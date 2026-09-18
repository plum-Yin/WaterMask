_base_ = './water_r101_fpn_3x_usis10k.py'

lr_config = dict(step=[8, 11])
runner = dict(max_epochs=12)
