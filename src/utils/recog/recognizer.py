import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel, BaseModule, Sequential
from mmengine.structures import LabelData
from mmengine.model import BaseDataPreprocessor, stack_batch
from mmaction.registry import MODELS
from mmaction.utils import register_all_modules

class DataPreprocessorZelda(BaseDataPreprocessor):
    def __init__(self, mean, std):
        super().__init__()

        self.register_buffer(
            'mean',
            torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1, 1),
            False)
        self.register_buffer(
            'std',
            torch.tensor(std, dtype=torch.float32).view(-1, 1, 1, 1),
            False)

    def forward(self, data, training=False):
        data = self.cast_data(data)
        inputs = data['inputs']
        batch_inputs = stack_batch(inputs)  # Batching
        batch_inputs = (batch_inputs - self.mean) / self.std  # Normalization
        data['inputs'] = batch_inputs
        return data

@MODELS.register_module()
class BackBoneZelda(BaseModule):
    def __init__(self, init_cfg=None):
        if init_cfg is None:
            init_cfg = [dict(type='Kaiming', layer='Conv3d', mode='fan_out', nonlinearity="relu"),
                        dict(type='Constant', layer='BatchNorm3d', val=1, bias=0)]

        super(BackBoneZelda, self).__init__(init_cfg=init_cfg)

        self.conv1 = Sequential(nn.Conv3d(3, 64, kernel_size=(3, 7, 7),
                                          stride=(1, 2, 2), padding=(1, 3, 3)),
                                nn.BatchNorm3d(64), nn.ReLU())
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                    padding=(0, 1, 1))

        self.conv = Sequential(nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
                               nn.BatchNorm3d(128), nn.ReLU())

    def forward(self, imgs):
        # imgs: [batch_size*num_views, 3, T, H, W]
        # features: [batch_size*num_views, 128, T/2, H//8, W//8]
        features = self.conv(self.maxpool(self.conv1(imgs)))
        return features


@MODELS.register_module()
class ClsHeadZelda(BaseModule):
    def __init__(self, num_classes, in_channels, dropout=0.5, average_clips='prob', init_cfg=None):
        if init_cfg is None:
            init_cfg = dict(type='Normal', layer='Linear', std=0.01)

        super(ClsHeadZelda, self).__init__(init_cfg=init_cfg)

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.average_clips = average_clips

        if dropout != 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.fc = nn.Linear(self.in_channels, self.num_classes)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        N, C, T, H, W = x.shape
        x = self.pool(x)
        x = x.view(N, C)
        assert x.shape[1] == self.in_channels

        if self.dropout is not None:
            x = self.dropout(x)

        cls_scores = self.fc(x)
        return cls_scores

    def loss(self, feats, data_samples):
        cls_scores = self(feats)
        labels = torch.stack([x.gt_labels.item for x in data_samples])
        labels = labels.squeeze()

        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)

        loss_cls = self.loss_fn(cls_scores, labels)
        return dict(loss_cls=loss_cls)

    def predict(self, feats, data_samples):
        cls_scores = self(feats)
        num_views = cls_scores.shape[0] // len(data_samples)
        # assert num_views == data_samples[0].num_clips
        cls_scores = self.average_clip(cls_scores, num_views)

        for ds, sc in zip(data_samples, cls_scores):
            pred = LabelData(item=sc)
            ds.pred_scores = pred
        return data_samples

    def average_clip(self, cls_scores, num_views):
          if self.average_clips not in ['score', 'prob', None]:
            raise ValueError(f'{self.average_clips} is not supported. '
                             f'Currently supported ones are '
                             f'["score", "prob", None]')

          total_views = cls_scores.shape[0]
          cls_scores = cls_scores.view(total_views // num_views, num_views, -1)

          if self.average_clips is None:
              return cls_scores
          elif self.average_clips == 'prob':
              cls_scores = F.softmax(cls_scores, dim=2).mean(dim=1)
          elif self.average_clips == 'score':
              cls_scores = cls_scores.mean(dim=1)

          return cls_scores


@MODELS.register_module()
class RecognizerZelda(BaseModel):
    def __init__(self, backbone, cls_head):
        super().__init__()

        self.backbone = MODELS.build(backbone)
        self.cls_head = MODELS.build(cls_head)

    def extract_feat(self, inputs):
        inputs = inputs.view((-1, ) + inputs.shape[2:])
        return self.backbone(inputs)

    def loss(self, inputs, data_samples):
        feats = self.extract_feat(inputs)
        loss = self.cls_head.loss(feats, data_samples)
        return loss

    def predict(self, inputs, data_samples):
        feats = self.extract_feat(inputs)
        predictions = self.cls_head.predict(feats, data_samples)
        return predictions

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if mode == 'tensor':
            return self.extract_feat(inputs)
        elif mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode: {mode}')