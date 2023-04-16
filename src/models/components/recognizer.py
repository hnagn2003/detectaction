import torch
import torch.nn as nn
from mmaction.registry import MODELS
from mmaction.utils import register_all_modules

register_all_modules(init_default_scope=True)
from src.utils.recog.recognizer import RecognizerZelda, BackBoneZelda, ClsHeadZelda #dont delete this

class SimpleRecog(nn.Module):
    def __init__(
        self,
        type='RecognizerZelda',
        type_backbone='BackBoneZelda',
        type_cls_head='ClsHeadZelda',
        num_classes=2,
        in_channels=128,
        average_clips='prob'
    ):
        super().__init__()
        backbone=dict(type=type_backbone),
        cls_head=dict(
            type=type_cls_head,
            num_classes=num_classes,
            in_channels=in_channels,
            average_clips=average_clips
        )
        model_cfg = dict(
            type=type,
            backbone=dict(type='BackBoneZelda'),
            cls_head=dict(
                type='ClsHeadZelda',
                num_classes=2,
                in_channels=128,
                average_clips='prob'))
        self.model = MODELS.build(model_cfg)
    def forward(self, x):
        return self.model.forward(x)
if __name__ == "__main__":
#     model = SimpleRecog("mmaction2/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x5-100e_kinetics400-rgb.py",
#                         'src/models/checkpoints/tsn_r50_video_1x1x8_100e_kinetics400_rgb_20200702-568cde33.pth',
#                         'cuda')
#     x = torch.randn(1,1,3,224,224)
#     device = torch.device('cuda')
#     x = x.to( torch.device(device))
# # print(x.shape)
#     output = model.forward(x)
#     print(output)
    # _ = SimpleRecog()
    
    _ = SimpleRecog()
    x = torch.randn(2,10,3,10,224,224) #batch frame channel segments H W
    
    # device = torch.device('cuda')
    # x = x.to( torch.device(device))
# print(x.shape)
    output = _.forward(x)
    print(output)
    print(output.shape)
    