_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py'
# _base_ = '../dcn/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py'

data_root = 'debug_data/icf_debug_data_root/'

model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)
    ),
    roi_head=dict(
        bbox_head=dict(num_classes=1),
    )
)

runner = dict(
    type='EpochBasedRunner',  
    max_epochs=36
    ) 

dataset_type = 'COCODataset'
classes = ('face', )
data = dict(
    samples_per_gpu = 8,
    workers_per_gpu = 8,
    train=dict(
        img_prefix= data_root + 'personai_icartoonface_detval',
        classes=classes,
        ann_file= data_root + 'val_0-1000.json'
    ),
    val=dict(
        img_prefix= data_root + 'personai_icartoonface_detval',
        classes=classes,
        ann_file= data_root + 'val_0-1000.json'
    ),
    test=dict(
        img_prefix= data_root + 'personai_icartoonface_detval',
        classes=classes,
        ann_file= data_root + 'val_0-1000.json'
    )
)

