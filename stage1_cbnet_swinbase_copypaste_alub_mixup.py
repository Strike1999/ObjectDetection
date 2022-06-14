model = dict(
    type='CascadeRCNN',
    pretrained="/farm/caixinhao/mmdetection/pretrained/swin_base_patch4_window7_224.pth",
    backbone=dict(
        type='CBSwinTransformer',
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False),
    neck=dict(
        type='CBFPN',
        in_channels=[128, 256, 512, 1024],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.25, 0.5, 1.0, 2.0, 4.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=50,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=50,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=50,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0))
        ],
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32])),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=5000,
            nms_post=5000,
            max_per_img=5000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.001),
            max_per_img=100)))
data_root = '../fewshotlogodetection_train/train/'
test_data_root = '../fewshotlogodetection_test/val/'
dataset_type = 'FewshotLogDet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=180,
        interpolation=1,
        p=0.3),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.3),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.1),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.1),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='MotionBlur', p=1.0),
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1)
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(2048, 800), (2048, 1500)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(
        type='Albu',
        transforms=[
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=180,
                interpolation=1,
                p=0.3),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.3),
            dict(
                type='OneOf',
                transforms=[
                    dict(
                        type='RGBShift',
                        r_shift_limit=10,
                        g_shift_limit=10,
                        b_shift_limit=10,
                        p=1.0),
                    dict(
                        type='HueSaturationValue',
                        hue_shift_limit=20,
                        sat_shift_limit=30,
                        val_shift_limit=20,
                        p=1.0)
                ],
                p=0.1),
            dict(
                type='JpegCompression',
                quality_lower=85,
                quality_upper=95,
                p=0.1),
            dict(type='ChannelShuffle', p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='MotionBlur', p=1.0),
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.1)
        ],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap=dict(img='image', gt_bboxes='bboxes'),
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='MixUp', p=0.5, lambd=0.5),
    dict(type='CutOut',
        n_holes=7,
        cutout_shape=[(4, 4), (4, 8), (8, 4),
                           (8, 8), (16, 8), (8, 16),
                           (16, 16), (16, 32), (32, 16) ]
    ),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(2048, 800), (2048, 1000), (2048, 1200), (2048, 1400),
                   (2048, 1600), (2048, 1800)],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
datasetA = dict(
    type='FewshotLogDet',
    ann_file=
    '/farm/caixinhao/data_split/split/annotations/instances_train2017_aug4.json',
    img_prefix='/farm/caixinhao/data_split/split/result_with_train_ann3/',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='Resize',
            img_scale=[(2048, 800), (2048, 1500)],
            multiscale_mode='range',
            keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(
            type='Albu',
            transforms=[
                dict(
                    type='ShiftScaleRotate',
                    shift_limit=0.0625,
                    scale_limit=0.0,
                    rotate_limit=180,
                    interpolation=1,
                    p=0.3),
                dict(
                    type='RandomBrightnessContrast',
                    brightness_limit=[0.1, 0.3],
                    contrast_limit=[0.1, 0.3],
                    p=0.3),
                dict(
                    type='OneOf',
                    transforms=[
                        dict(
                            type='RGBShift',
                            r_shift_limit=10,
                            g_shift_limit=10,
                            b_shift_limit=10,
                            p=1.0),
                        dict(
                            type='HueSaturationValue',
                            hue_shift_limit=20,
                            sat_shift_limit=30,
                            val_shift_limit=20,
                            p=1.0)
                    ],
                    p=0.1),
                dict(
                    type='JpegCompression',
                    quality_lower=85,
                    quality_upper=95,
                    p=0.1),
                dict(type='ChannelShuffle', p=0.1),
                dict(
                    type='OneOf',
                    transforms=[
                        dict(type='MotionBlur', p=1.0),
                        dict(type='Blur', blur_limit=3, p=1.0),
                        dict(type='MedianBlur', blur_limit=3, p=1.0)
                    ],
                    p=0.1)
            ],
            bbox_params=dict(
                type='BboxParams',
                format='pascal_voc',
                label_fields=['gt_labels'],
                min_visibility=0.0,
                filter_lost_elements=True),
            keymap=dict(img='image', gt_bboxes='bboxes'),
            update_pad_shape=False,
            skip_img_without_anno=True),
        dict(type='MixUp', p=0.5, lambd=0.5),
        dict(type='CutOut',
             n_holes=7,
             cutout_shape=[(4, 4), (4, 8), (8, 4),
                           (8, 8), (16, 8), (8, 16),
                           (16, 16), (16, 32), (32, 16)]
             ),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ])
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='ClassBalancedDataset',
        oversample_thr=0.002,
        dataset=dict(
            type='RepeatDataset',
            times=1,
            dataset=dict(
                type='ConcatDataset',
                datasets=[
                    dict(
                        type='FewshotLogDet',
                        ann_file=
                        '/farm/caixinhao/data_split/split/annotations/instances_train2017_aug4.json',
                        img_prefix=
                        '/farm/caixinhao/data_split/split/result_with_train_ann3/',
                        pipeline=[
                            dict(type='LoadImageFromFile'),
                            dict(type='LoadAnnotations', with_bbox=True),
                            dict(
                                type='Resize',
                                img_scale=[(2048, 800), (2048, 1500)],
                                multiscale_mode='range',
                                keep_ratio=True),
                            dict(type='RandomFlip', flip_ratio=0.5),
                            dict(
                                type='Normalize',
                                mean=[123.675, 116.28, 103.53],
                                std=[58.395, 57.12, 57.375],
                                to_rgb=True),
                            dict(
                                type='Albu',
                                transforms=[
                                    dict(
                                        type='ShiftScaleRotate',
                                        shift_limit=0.0625,
                                        scale_limit=0.0,
                                        rotate_limit=180,
                                        interpolation=1,
                                        p=0.3),
                                    dict(
                                        type='RandomBrightnessContrast',
                                        brightness_limit=[0.1, 0.3],
                                        contrast_limit=[0.1, 0.3],
                                        p=0.3),
                                    dict(
                                        type='OneOf',
                                        transforms=[
                                            dict(
                                                type='RGBShift',
                                                r_shift_limit=10,
                                                g_shift_limit=10,
                                                b_shift_limit=10,
                                                p=1.0),
                                            dict(
                                                type='HueSaturationValue',
                                                hue_shift_limit=20,
                                                sat_shift_limit=30,
                                                val_shift_limit=20,
                                                p=1.0)
                                        ],
                                        p=0.1),
                                    dict(
                                        type='JpegCompression',
                                        quality_lower=85,
                                        quality_upper=95,
                                        p=0.1),
                                    dict(type='ChannelShuffle', p=0.1),
                                    dict(
                                        type='OneOf',
                                        transforms=[
                                            dict(type='MotionBlur', p=1.0),
                                            dict(
                                                type='Blur',
                                                blur_limit=3,
                                                p=1.0),
                                            dict(
                                                type='MedianBlur',
                                                blur_limit=3,
                                                p=1.0)
                                        ],
                                        p=0.1)
                                ],
                                bbox_params=dict(
                                    type='BboxParams',
                                    format='pascal_voc',
                                    label_fields=['gt_labels'],
                                    min_visibility=0.0,
                                    filter_lost_elements=True),
                                keymap=dict(img='image', gt_bboxes='bboxes'),
                                update_pad_shape=False,
                                skip_img_without_anno=True),
                            dict(type='MixUp', p=0.5, lambd=0.5),
                            dict(type='CutOut',
                                 n_holes=7,
                                 cutout_shape=[(4, 4), (4, 8), (8, 4),
                                               (8, 8), (16, 8), (8, 16),
                                               (16, 16), (16, 32), (32, 16)]
                                 ),
                            dict(type='Pad', size_divisor=32),
                            dict(type='DefaultFormatBundle'),
                            dict(
                                type='Collect',
                                keys=['img', 'gt_bboxes', 'gt_labels'])
                        ])
                ]))),
    val=dict(
        type='FewshotLogDet',
        ann_file=
        '/farm/caixinhao/fewshotlogodetection_train/train/annotations/instances_val2017_split.json',
        img_prefix='../fewshotlogodetection_train/train/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=[(2048, 800), (2048, 1000), (2048, 1200),
                           (2048, 1400), (2048, 1600), (2048, 1800)],
                flip=True,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='FewshotLogDet',
        ann_file=
        '../fewshotlogodetection_test/val/annotations/instances_val2017.json',
        img_prefix='../fewshotlogodetection_test/val/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=[(2048, 800), (2048, 1000), (2048, 1200),
                           (2048, 1400), (2048, 1600), (2048, 1800)],
                flip=True,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=30, metric='bbox')
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict(
    grad_clip=None,
    type='DistOptimizerHook',
    update_interval=1,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[21, 25, 27])
runner = dict(type='EpochBasedRunnerAmp', max_epochs=30)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
fp16 = None
work_dir = './work_dirs/bisai/swin_base/cascade_mask_rcnn'
gpu_ids = range(0, 4)
