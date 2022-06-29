_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py'

data_root = "/workspace/mmdet-anime-face/icartoonface_coco_format/"

model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)
        ),
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        ))

runner = dict(
    type='EpochBasedRunner',  
    max_epochs=36
    ) 

dataset_type = 'COCODataset'
classes = ('face', )
data = dict(
    samples_per_gpu = 8,
    workers_per_gpu = 10,
    train=dict(
        img_prefix= data_root + 'icartoonface_dettrain',
        classes=classes,
        ann_file=data_root + 'train.json'
    ),
    val=dict(
        img_prefix= data_root + 'personai_icartoonface_detval',
        classes=classes,
        ann_file=data_root + 'val.json'
    ),
    test=dict(
        img_prefix= data_root + 'personai_icartoonface_detval',
        classes=classes,
        ann_file=data_root + 'val.json'
    )
)
