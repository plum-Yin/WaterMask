_base_ = './water_r101_fpn_3x_usis10k.py'

num_classes = 1
classes = ('foreground',)
data_root = '../ViT-UWA/datasets/USIS10K/'

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=num_classes),
        mask_head=dict(
            classes_num_in_stages=[num_classes, num_classes, num_classes])))

data = dict(
    train=dict(
        classes=classes,
        ann_file=(data_root +
                  'foreground_annotations/foreground_train_annotations.json')),
    val=dict(
        classes=classes,
        ann_file=(data_root +
                  'foreground_annotations/foreground_val_annotations.json')),
    test=dict(
        classes=classes,
        ann_file=(data_root +
                  'foreground_annotations/foreground_test_annotations.json')))
