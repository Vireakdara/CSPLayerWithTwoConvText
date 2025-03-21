@MODELS.register_module()
class CSPLayerWithTwoConvText(BaseModule):
    """CSP Layer with 2 convs + optional iterative cross-attention + FiLM gating."""
    
    def __init__(self, in_channels: int, out_channels: int, expand_ratio: float = 0.5,
                 num_blocks: int = 1, add_identity: bool = True, guide_channels: int = 256,
                 embed_channels: int = 512, num_heads: int = 4, num_fusion_stages: int = 1,
                 use_film: bool = False, conv_cfg: Optional[dict] = None, norm_cfg: dict = dict(type='BN'),
                 act_cfg: dict = dict(type='SiLU', inplace=True), init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)

        self.mid_channels = int(out_channels * expand_ratio)
        self.num_fusion_stages = num_fusion_stages
        self.use_film = use_film

        # Main 1x1 conv that splits into two mid_channels
        self.main_conv = ConvModule(in_channels, 2 * self.mid_channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # Darknet Bottleneck blocks
        self.blocks = nn.ModuleList([DarknetBottleneck(self.mid_channels, self.mid_channels, expansion=1,
                                                      kernel_size=(3, 3), padding=(1, 1), add_identity=add_identity)
                                    for _ in range(num_blocks)])

        # Final 1x1 conv after concatenation
        self.final_conv = ConvModule((2 + num_blocks) * self.mid_channels, out_channels, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # Text fusion modules (cross-attention + optional FiLM)
        self.img_proj = nn.Conv2d(self.mid_channels, guide_channels, kernel_size=1)
        # Correctly initialize text projection with embed_channels (fixed during __init__)
        self.txt_proj = nn.Linear(embed_channels, guide_channels)
        self.cross_attn = nn.MultiheadAttention(embed_dim=guide_channels, num_heads=num_heads, batch_first=True)
        if self.use_film:
            self.film = FiLMLayer(txt_dim=embed_channels, num_channels=self.mid_channels)

    def forward(self, x: torch.Tensor, txt_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Step A: Split features
        x_main = self.main_conv(x)
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), dim=1)

        # Step B: Optionally fuse text embeddings into chunk2
        if txt_feats is not None:
            if txt_feats.dim() == 3:  # [B, L, embed_channels]
                txt_emb = txt_feats.mean(dim=1)  # [B, embed_channels]
            else:
                txt_emb = txt_feats  # [B, embed_channels]

            # Apply text projection (pre-defined during __init__)
            txt_proj = self.txt_proj(txt_emb)  # [B, guide_channels]

            # Step B2: Apply FiLM gating (optional)
            if self.use_film:
                x_main[1] = self.film(x_main[1], txt_emb)

            # Step B3: Cross-Attention Fusion
            B, C, H, W = x_main[1].shape
            img_tokens = self.img_proj(x_main[1])  # [B, guide_channels, H, W]
            img_tokens = img_tokens.flatten(2).transpose(1, 2)  # [B, H*W, guide_channels]

            txt_proj = txt_proj.unsqueeze(1)  # [B, 1, guide_channels]
            attn_out, _ = self.cross_attn(query=img_tokens, key=txt_proj, value=txt_proj)
            attn_out = attn_out.transpose(1, 2).view(B, -1, H, W)  # [B, guide_channels, H, W]

            # Merge attention output with image features
            x_main[1] = x_main[1] + attn_out

        # Step C: Process through DarknetBottleneck blocks
        for block in self.blocks:
            x_main.append(block(x_main[-1]))

        # Step D: Concatenate and final conv
        out = torch.cat(x_main, dim=1)
        out = self.final_conv(out)
        return out
