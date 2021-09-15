# dataset settings
dataset_type = 'DrivingStereo'
data_root = 'data/driving_stereo/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (576, 384)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDepth'),
    dict(type='Resize', img_scale=(4096, 2048), ratio_range=(0.5, 2.0)),
    dict(type='DepthRandomCrop', crop_size=crop_size, valid_min_ratio=0.1),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'depth_map']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=crop_size,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img', 'depth_map']),
            dict(type='Collect', keys=['img', 'depth_map']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='left/',
        depth_map_dir='depth_maps/',
        img_suffix='.jpg',
        depth_map_suffix='.png',
        pipeline=train_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        depth_map_dir='depth_maps/',
        img_suffix='.jpg',
        depth_map_suffix='.png',
        pipeline=test_pipeline)
)
