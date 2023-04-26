import torch
import torch.nn as nn
from mmaction.registry import MODELS
from mmaction.utils import register_all_modules
from mmengine import Config
from lightning import LightningModule, LightningDataModule


register_all_modules(init_default_scope=True)
from mmengine.model import BaseDataPreprocessor, stack_batch

from src.utils.recog.recognizer import RecognizerZelda, BackBoneZelda, ClsHeadZelda, DataPreprocessorZelda #dont delete this


class SimpleRecog(nn.Module):
    def __init__(
        self,
        config_file : str = 'mmaction2/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py',
        num_classes : int = 2,
        checkpoint_file : str = "tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth"
    ):
        super().__init__()
        self.config = Config.fromfile(config_file)
        self.config.model.cls_head.num_classes = num_classes
        self.checkpoint_file = checkpoint_file
        if checkpoint_file is not None :
            self.config.load_from = checkpoint_file
        self.model = MODELS.build(self.config.model)
        

    def forward(self, x):
        # _ = (tensor.to('cpu') for tensor in x["inputs"])
        # self.model.to('cpu')
        device = next(self.model.parameters())
        data_preprocessor_cfg = dict(
        type='DataPreprocessorZelda',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375])
        self.model.data_preprocessor = MODELS.build(data_preprocessor_cfg)
        x = self.model.data_preprocessor(x, training=True)
        # _ = (tensor.to(device) for tensor in x["inputs"])
        # print(x["inputs"].device)
        # self.model.to(device)
        # x = [tensor.to(device) for tensor in x["inputs"]]
        # output = self.model.forward(x["inputs"],x["data_samples"],mode = "predict") # bug
        print(device)
        if (device.is_cuda):
            # for key in x:
            #     x[key] = x[key].to('cuda')
            x = recursive_to_device(x, 'cuda')
        
        output = self.model(**x, mode='predict')
        data_samples = [d.to_dict() for d in output]
        pred = [d['pred_scores']['item'] for d in data_samples]
        y = [d['gt_labels']['item'] for d in data_samples]
        # stack all the preds to 1 tensor
        pred = torch.stack(pred)
        # [tensor([1]), tensor([1])] -> tensor([1, 1])
        y = torch.tensor(y)
        return pred
    

def recursive_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: recursive_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [recursive_to_device(x, device) for x in data]
    else:
        return data
    
if __name__ == "__main__":
    import pyrootutils
    from omegaconf import DictConfig
    import hydra
    import numpy as np
    from PIL import Image, ImageDraw
    from tqdm import tqdm


    path = pyrootutils.find_root(
        search_from=__file__, indicator=".project-root")
    config_path = str(path / "configs" / "data")
    output_path = path / "outputs"
    print("root", path, config_path)
    # pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


    def test_datamodule(cfg: DictConfig):
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg)
        model = SimpleRecog()
        
        print(datamodule)
        datamodule.prepare_data()
        datamodule.setup()
        loader = datamodule.train_dataloader()
        # # loader
        batch = next(iter(loader))
        y = batch['data_samples']
        data_samples = [d.to_dict() for d in y]
        y = [d['gt_labels']['item'] for d in data_samples]
        y = torch.tensor(y)
        criterion = torch.nn.CrossEntropyLoss()
        output = model(batch)
        loss = criterion(output, y)
        preds = torch.argmax(output, dim=1)
        print(preds)
        print(y)
        print("******************************************")
        print(loss)

    @hydra.main(version_base="1.3", config_path=config_path, config_name="recog.yaml")
    def main(cfg: DictConfig):
        print(cfg)
        test_datamodule(cfg)

    main()
    