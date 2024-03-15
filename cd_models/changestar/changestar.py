import ever as er
import torch
import torch.nn as nn
from ever.module import fpn
from albumentations import Compose, OneOf, Normalize
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, RandomCrop

from .mixin import ChangeMixin
from .segmentation import Segmentation


@er.registry.MODEL.register()
class ChangeStar(er.ERModule):
    '''
@inproceedings{zheng2021change,
  title={Change is Everywhere: Single-Temporal Supervised Object Change Detection in Remote Sensing Imagery},
  author={Zheng, Zhuo and Ma, Ailong and Zhang, Liangpei and Zhong, Yanfei},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={15193--15202},
  year={2021}
}

@inproceedings{zheng2020foreground,
  title={Foreground-Aware Relation Network for Geospatial Object Segmentation in High Spatial Resolution Remote Sensing Imagery},
  author={Zheng, Zhuo and Zhong, Yanfei and Wang, Junjue and Ma, Ailong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4096--4105},
  year={2020}
}'''
    def __init__(self, config):
        super().__init__(config)

        segmentation = Segmentation(self.config.segmenation)

        layers = [nn.Conv2d(self.config.classifier.in_channels, self.config.classifier.out_channels, 3, 1, 1),
                  nn.UpsamplingBilinear2d(scale_factor=self.config.classifier.scale)]
        classifier = nn.Sequential(*layers)

        self.changemixin = ChangeMixin(segmentation, classifier, self.config.detector, self.config.loss_config)

    def forward(self, x, y=None):
        x = torch.cat([x, y], 1)
        y = None
        if self.training or x.size(1) == 6:
            # segmentation + change detection
            return self.changemixin(x, y)

        if x.size(1) == 3:
            # only segmentation
            seg_logit = self.changemixin.classify(self.changemixin.extract_feature(x))
            return seg_logit.sigmoid()

    def set_default_config(self):
        self.config.update(dict(
            segmenation=dict(),
            classifier=dict(
                in_channels=256,
                out_channels=2,
                scale=4.0
            ),
            detector=dict(
                name='convs',
                in_channels=256 * 2,
                inner_channels=16,
                out_channels=1,
                num_convs=4,
            ),
            loss_config=dict(
                semantic=dict(ignore_index=-1),
                change=dict(ignore_index=-1)
            )
        ))

    # def log_info(self):
    #     return dict(
    #         cfg=self.config
    #     )

data = dict(
    train=dict(
        type='LEVIRCDLoader',
        params=dict(
            root_dir=(
                './LEVIR-CD/train',
            ),
            transforms=Compose([
                OneOf([
                    HorizontalFlip(True),
                    VerticalFlip(True),
                    RandomRotate90(True)
                ], p=0.5),
                er.preprocess.albu.RandomDiscreteScale([0.75, 1.25, 1.5], p=0.5),
                RandomCrop(512, 512, True),
                Normalize(mean=(0.485, 0.456, 0.406, 0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225, 0.229, 0.224, 0.225), max_pixel_value=255),
                er.preprocess.albu.ToTensor(),
            ]),
            batch_size=16,
            num_workers=8,
            training=True
        ),
    ),
    test=dict(
        type='LEVIRCDLoader',
        params=dict(
            root_dir='./LEVIR-CD/test',
            transforms=Compose([
                Normalize(mean=(0.485, 0.456, 0.406, 0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225, 0.229, 0.224, 0.225), max_pixel_value=255),
                er.preprocess.albu.ToTensor(),
            ]),
            batch_size=4,
            num_workers=0,
            training=False
        ),
    ),
)
optimizer = dict(
    type='sgd',
    params=dict(
        momentum=0.9,
        weight_decay=0.0001
    ),
    grad_clip=dict(
        max_norm=35,
        norm_type=2,
    )
)
learning_rate = dict(
    type='poly',
    params=dict(
        base_lr=0.03,
        power=0.9,
        max_iters=5600,
    )
)
train = dict(
    forward_times=1,
    num_iters=100,
    eval_per_epoch=False,
    summary_grads=False,
    summary_weights=False,
    distributed=True,
    apex_sync_bn=True,
    sync_bn=False,
    eval_after_train=True,
    log_interval_step=100,
    save_ckpt_interval_epoch=10,
    eval_interval_epoch=1,
)
test = dict(
)

config = dict(
    model=dict(
        type='ChangeStarBiSup',
        params=dict(
            segmenation=dict(
                model_type='farseg',
                backbone=dict(
                    resnet_type='resnet50',
                    pretrained=True,
                    freeze_at=0,
                    output_stride=32,
                ),
                head=dict(
                    fpn=dict(
                        in_channels_list=(256, 512, 1024, 2048),
                        out_channels=256,
                        conv_block=fpn.conv_bn_relu_block
                    ),
                    fs_relation=dict(
                        scene_embedding_channels=2048,
                        in_channels_list=(256, 256, 256, 256),
                        out_channels=256,
                        scale_aware_proj=True
                    ),
                    fpn_decoder=dict(
                        in_channels=256,
                        out_channels=256,
                        in_feat_output_strides=(4, 8, 16, 32),
                        out_feat_output_stride=4,
                        classifier_config=None
                    )
                ),
            ),
            detector=dict(
                name='convs',
                in_channels=256 * 2,
                inner_channels=16,
                out_channels=1,
                scale=4.0,
                num_convs=4,
            ),
            loss_config=dict(
                bce=True,
                dice=True,
                ignore_index=-1
            )
        )
    ),
    data=data,
    optimizer=optimizer,
    learning_rate=learning_rate,
    train=train,
    test=test
)

def ChangeStar_R50(conf = config['model']['params']):
    # print(conf['segmenation']['model_type'])
    return ChangeStar(conf)
