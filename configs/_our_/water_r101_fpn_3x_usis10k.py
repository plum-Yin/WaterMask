_base_ = './water_r50_fpn_1x.py'

num_classes = 7
classes = (
    'wrecks/ruins',
    'fish',
    'reefs',
    'aquatic plants',
    'human divers',
    'robots',
    'sea-floor',
)
data_root = '../ViT-UWA/datasets/USIS10K/'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(
            type='Pretrained', checkpoint='torchvision://resnet101')),
    roi_head=dict(
        bbox_head=dict(num_classes=num_classes),
        mask_head=dict(
            classes_num_in_stages=[num_classes, num_classes, 1])))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    persistent_workers=True,
    train=dict(
        _delete_=True,
        type='CocoDataset',
        classes=classes,
        ann_file=(data_root +
                  'multi_class_annotations/multi_class_train_annotations.json'),
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline),
    val=dict(
        _delete_=True,
        type='CocoDataset',
        samples_per_gpu=2,
        classes=classes,
        ann_file=(data_root +
                  'multi_class_annotations/multi_class_val_annotations.json'),
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline),
    test=dict(
        _delete_=True,
        type='CocoDataset',
        samples_per_gpu=2,
        classes=classes,
        ann_file=(data_root +
                  'multi_class_annotations/multi_class_test_annotations.json'),
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline))

optimizer = dict(
    type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[27, 33],
    gamma=0.1)
runner = dict(type='EpochBasedRunner', max_epochs=36)

evaluation = dict(
    interval=1,
    metric=['bbox', 'segm'],
    classwise=True,
    save_best='segm_mAP',
    rule='greater')
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
