import torch
import torch.nn as nn
from typing import Optional, Sequence
from mmengine.model import BaseModule
from mmyolo.registry import MODELS
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.layers.csp_layer import \
    DarknetBottleneck as MMDET_DarknetBottleneck

# Optional: A simple FiLM gating layer
class FiLMLayer(nn.Module):
    """Produce gamma, beta from text embeddings to modulate image features."""
    def __init__(self, txt_dim, num_channels):
        super().__init__()
        # Linear to produce (gamma, beta) for each channel
        self.film_gen = nn.Linear(txt_dim, 2 * num_channels, bias=True)

    def forward(self, feats: torch.Tensor, txt_emb: torch.Tensor) -> torch.Tensor:
        """
        feats: B x C x H x W
        txt_emb: B x txt_dim (pooled text embedding)
        """
        B, C, H, W = feats.shape
        film_params = self.film_gen(txt_emb)  # (B x 2*C)
        gamma, beta = film_params[:, :C], film_params[:, C:]
        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)
        return feats * (1 + gamma) + beta
    
class DarknetBottleneck(MMDET_DarknetBottleneck):
    """The basic bottleneck block used in Darknet.

    Each ResBlock consists of two ConvModules and the input is added to the
    final output. Each ConvModule is composed of Conv, BN, and LeakyReLU.
    The first convLayer has filter size of k1Xk1 and the second one has the
    filter size of k2Xk2.

    Note:
    This DarknetBottleneck is little different from MMDet's, we can
    change the kernel size and padding for each conv.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        expansion (float): The kernel size for hidden channel.
            Defaults to 0.5.
        kernel_size (Sequence[int]): The kernel size of the convolution.
            Defaults to (1, 3).
        padding (Sequence[int]): The padding size of the convolution.
            Defaults to (0, 1).
        add_identity (bool): Whether to add identity to the out.
            Defaults to True
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Defaults to False
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='Swish').
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion: float = 0.5,
                 kernel_size: Sequence[int] = (1, 3),
                 padding: Sequence[int] = (0, 1),
                 add_identity: bool = True,
                 use_depthwise: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(in_channels, out_channels, init_cfg=init_cfg)
        hidden_channels = int(out_channels * expansion)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        assert isinstance(kernel_size, Sequence) and len(kernel_size) == 2

        self.conv1 = ConvModule(
            in_channels,
            hidden_channels,
            kernel_size[0],
            padding=padding[0],
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = conv(
            hidden_channels,
            out_channels,
            kernel_size[1],
            stride=1,
            padding=padding[1],
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.add_identity = \
            add_identity and in_channels == out_channels

@MODELS.register_module()
class CSPLayerWithTwoConvText(BaseModule):
    """CSP Layer with 2 convs + cross-attention + FiLM gating."""
    
    def __init__(self, in_channels: int, out_channels: int, expand_ratio: float = 0.5, 
                 mid_channels: int = 256,  # Explicit mid_channels control
                 num_blocks: int = 1, add_identity: bool = True, guide_channels: int = 256,
                 embed_channels: int = 512, num_heads: int = 8,  # 256//32=8 heads
                 num_fusion_stages: int = 1, use_film: bool = False, 
                 conv_cfg: Optional[dict] = None, 
                 norm_cfg: dict = dict(type='BN'), 
                 act_cfg: dict = dict(type='SiLU', inplace=True),
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)

        self.mid_channels = mid_channels
        self.guide_channels = guide_channels
        self.num_fusion_stages = num_fusion_stages
        self.use_film = use_film

        # Main 1x1 conv splits features
        self.main_conv = ConvModule(in_channels, 2 * self.mid_channels, 1,
                                   conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # Darknet bottlenecks
        self.blocks = nn.ModuleList([
            DarknetBottleneck(self.mid_channels, self.mid_channels, expansion=1,
                              kernel_size=(3, 3), padding=(1, 1), add_identity=add_identity)
            for _ in range(num_blocks)
        ])

        # Final conv after concatenation
        self.final_conv = ConvModule((2 + num_blocks) * self.mid_channels, out_channels, 1,
                                    conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # Text fusion modules
        self.img_proj = nn.Conv2d(self.mid_channels, self.guide_channels, kernel_size=1)
        self.txt_proj = nn.Linear(embed_channels, self.guide_channels)
        self.cross_attn = nn.MultiheadAttention(self.guide_channels, num_heads, batch_first=True)
        self.attn_proj = nn.Conv2d(self.guide_channels, self.mid_channels, kernel_size=1)
        
        if use_film:
            self.film = FiLMLayer(embed_channels, self.mid_channels)

    def forward(self, x: torch.Tensor, txt_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_main = self.main_conv(x)
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), dim=1))

        if txt_feats is not None:
            # Reduce text features
            if txt_feats.dim() == 3:
                txt_emb = txt_feats.mean(dim=1)  # [B, 512]
            else:
                txt_emb = txt_feats  # [B, 512]
            
            # Project text embeddings
            # print("txt_emb shape :", txt_emb.shape)
            txt_proj = self.txt_proj(txt_emb)  # [B, 256]
            
            if self.use_film:
                x_main[1] = self.film(x_main[1], txt_emb)

            # Cross-attention fusion
            B, C, H, W = x_main[1].shape
            img_tokens = self.img_proj(x_main[1])  # [B, 256->512, H, W]
            img_tokens = img_tokens.flatten(2).transpose(1, 2)  # [B, H*W, 512]
            
            txt_proj = txt_proj.unsqueeze(1)  # [B, 1, 512]
            attn_out, _ = self.cross_attn(img_tokens, txt_proj, txt_proj)
            attn_out = attn_out.transpose(1, 2).view(B, self.guide_channels, H, W)
            attn_out = self.attn_proj(attn_out)  # [B, 256, H, W]
            
            x_main[1] = x_main[1] + attn_out  # Both 256 channels

        # Process through blocks
        for block in self.blocks:
            x_main.append(block(x_main[-1]))

        out = torch.cat(x_main, dim=1)
        return self.final_conv(out)
