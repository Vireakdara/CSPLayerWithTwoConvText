_base_ = ('/home/wrf/Dara/YOLO-World/third_party/mmyolo/configs/yolov8/yolov8_l_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(imports=['yolo_world'], allow_failed_imports=False)


classnames = [["person"], ["bicycle"], ["car"], ["motorcycle"], ["airplane"], ["bus"], ["train"], ["truck"], ["boat"], ["traffic light"], ["fire hydrant"], ["stop sign"], ["parking meter"], ["bench"], ["bird"], ["cat"], ["dog"], ["horse"], ["sheep"], ["cow"], ["elephant"], ["bear"], ["zebra"], ["giraffe"], ["backpack"], ["umbrella"], ["handbag"], ["tie"], ["suitcase"], ["frisbee"], ["skis"], ["snowboard"], ["sports ball"], ["kite"], ["baseball bat"], ["baseball glove"], ["skateboard"], ["surfboard"], ["tennis racket"], ["bottle"], ["wine glass"], ["cup"], ["fork"], ["knife"], ["spoon"], ["bowl"], ["banana"], ["apple"], ["sandwich"], ["orange"], ["broccoli"], ["carrot"], ["hot dog"], ["pizza"], ["donut"], ["cake"], ["chair"], ["couch"], ["potted plant"], ["bed"], ["dining table"], ["toilet"], ["tv"], ["laptop"], ["mouse"], ["remote"], ["keyboard"], ["cell phone"], ["microwave"], ["oven"], ["toaster"], ["sink"], ["refrigerator"], ["book"], ["clock"], ["vase"], ["scissors"], ["teddy bear"], ["hair drier"], ["toothbrush"]]


# hyper-parameters
num_classes = 80
num_training_classes = 80
max_epochs = 40  # Maximum training epochs
close_mosaic_epochs = 30
save_epoch_intervals = 5
text_channels = 512
neck_embed_channels = [512, 512, 512]
neck_num_heads = [8, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 1e-3 # O 2e-4 1e-3 , 1e-5 0.0002
weight_decay = 0.0005 # 0.0005 0.05
train_batch_size_per_gpu = 16  # 16
load_from = 'pretrained_models/yolo_world_l_clip_t2i_bn_2e-3adamw_32xb16-100e_obj365v1_goldg_cc3mlite_train-ca93cd1f.pth'
text_model_name = 'pretrained/clip-vit-base-patch32'
# text_model_name = 'beit-base-patch16-224'
# text_model_name = 'xlm-align-base'  
# text_model_name = 'blip-image-captioning-large'
persistent_workers = False
# blip-image-captioning-large, xlm-align-base
# model settings
model = dict(type='YOLOWorldDetector',
             mm_neck=True,
             num_train_classes=num_training_classes,
             num_test_classes=num_classes,
             data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),
            #  freeze_prompt=False,
             backbone=dict(_delete_=True,
                           type='MultiModalYOLOBackbone',
                           image_model={{_base_.model.backbone}},
                           text_model=dict(type='HuggingCLIPLanguageBackbone', # HuggingCLIPCocoOpLanguageBackbone HuggingALIGNLanguageBackbone HuggingCLIPLanguageBackbone HuggingEnALIGNLanguageBackbone HuggingBLIP2LanguageBackbone HuggingALIGNLanguageBackboneWithPrompts HuggingAltCLIPLanguageBackboneWithPrompts, HuggingAltCLIPLanguageBackbone, HuggingALIGNLanguageBackbone, # HuggingSBERTLanguageBackbone # EnhancedTextCLIPBackbone, HuggingBeitImageBackbone , HuggingCLIPLanguageBackbone
                                           model_name=text_model_name,
                                        #    classnames = 'data/texts/coco_class_texts.json',
                                           frozen_modules=['all'])),
             neck=dict(type='YOLOWorldPAFPN',
                       guide_channels=text_channels,
                       embed_channels=neck_embed_channels,
                       num_heads=neck_num_heads,
                       block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv')),
             bbox_head=dict(type='YOLOWorldHead',
                            head_module=dict(
                                type='YOLOWorldHeadModule',
                                use_bn_head=True,
                                embed_dims=text_channels,
                                num_classes=num_training_classes)),
             train_cfg=dict(assigner=dict(num_classes=num_training_classes)))

# dataset settings
text_transform = [
    dict(type='RandomLoadText',
         num_neg_samples=(num_classes, num_classes),
         max_num_samples=num_training_classes,
         padding_to_max=True,
         padding_value=''),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts'))
]
mosaic_affine_transform = [
    dict(type='MultiModalMosaic',
         img_scale=_base_.img_scale,
         pad_val=114.0,
         pre_transform=_base_.pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_aspect_ratio=100.,
        scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),
        # img_scale is (width, height)
        border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),
        border_val=(114, 114, 114))
]

train_pipeline = [
    *_base_.pre_transform, *mosaic_affine_transform,
    dict(type='YOLOv5MultiModalMixUp',
         prob=_base_.mixup_prob,
         pre_transform=[*_base_.pre_transform, *mosaic_affine_transform]),
    *_base_.last_transform[:-1], *text_transform
]
train_pipeline_stage2 = [*_base_.train_pipeline_stage2[:-1], *text_transform]

coco_train_dataset = dict(_delete_=True,
                          type='MultiModalDataset',
                          dataset=dict(
                              type='YOLOv5CocoDataset',
                              data_root='/home/wrf/Dara/coco2017label/coco/',
                              ann_file='annotations/instances_train2017.json',
                              data_prefix=dict(img='images/train2017/'),
                              filter_cfg=dict(filter_empty_gt=False,
                                              min_size=32)),
                          class_text_path='data/texts/coco_class_texts.json',
                          pipeline=train_pipeline)

train_dataloader = dict(persistent_workers=persistent_workers,
                        batch_size=train_batch_size_per_gpu,
                        collate_fn=dict(type='yolow_collate'),
                        dataset=coco_train_dataset)
test_pipeline = [
    *_base_.test_pipeline[:-1],
    dict(type='LoadText'),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param', 'texts'))
]
coco_val_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(type='YOLOv5CocoDataset',
                 data_root='/home/wrf/Dara/coco2017label/coco/',
                 ann_file='annotations/instances_val2017.json',
                 data_prefix=dict(img='images/val2017/'),
                 filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/coco_class_texts.json',
    pipeline=test_pipeline)
val_dataloader = dict(dataset=coco_val_dataset)
test_dataloader = val_dataloader
# training settings
default_hooks = dict(param_scheduler=dict(scheduler_type='linear',
                                          lr_factor=0.01,
                                          max_epochs=max_epochs),
                     checkpoint=dict(max_keep_ckpts=-1,
                                     save_best=None,
                                     interval=save_epoch_intervals))
custom_hooks = [
    dict(type='EMAHook',
         ema_type='ExpMomentumEMA',
         momentum=0.0001,
         update_buffers=True,
         strict_load=False,
         priority=49),
    dict(type='mmdet.PipelineSwitchHook',
         switch_epoch=max_epochs - close_mosaic_epochs,
         switch_pipeline=train_pipeline_stage2)
]
train_cfg = dict(max_epochs=max_epochs,
                 val_interval=1,
                 dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                                     _base_.val_interval_stage2)])
# optim_wrapper = dict(optimizer=dict(
#     _delete_=True,
#     type='SGD', #SGD
#     lr=base_lr,
#     momentum=0.937,
#     nesterov=True,
#     weight_decay=weight_decay,
#     batch_size_per_gpu=train_batch_size_per_gpu),
#                      paramwise_cfg=dict(
#                          custom_keys={
#                              'backbone.text_model': dict(lr_mult=0.01),
#                              'logit_scale': dict(weight_decay=0.0)
#                          }),
#                      constructor='YOLOWv5OptimizerConstructor')


optim_wrapper = dict(optimizer=dict(
    _delete_=True,
    type='AdamW', #SGD AdamW
    lr=base_lr,
    # momentum=0.937,
    # nesterov=True,
    weight_decay=weight_decay,
    batch_size_per_gpu=train_batch_size_per_gpu),
                     paramwise_cfg=dict(
                         bypass_duplicate=True,
                         custom_keys={
                             'backbone.text_model': dict(lr_mult=0.01),
                             'logit_scale': dict(weight_decay=0.0)
                         }),
                     constructor='YOLOWv5OptimizerConstructor')
# DefaultOptimizerConstructor
# evaluation settings
val_evaluator = dict(_delete_=True,
                     type='mmdet.CocoMetric',
                     proposal_nums=(100, 1, 10),
                     ann_file='/home/wrf/Dara/coco2017label/coco/annotations/instances_val2017.json',
                     metric='bbox')
