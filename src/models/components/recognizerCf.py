import torch
import torch.nn as nn
from mmaction.registry import MODELS
from mmaction.utils import register_all_modules
from mmengine import Config
from lightning import LightningModule, LightningDataModule


register_all_modules(init_default_scope=True)
from src.utils.recog.recognizer import RecognizerZelda, BackBoneZelda, ClsHeadZelda #dont delete this

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
        if checkpoint_file is not None :
            self.config.load_from = checkpoint_file
        self.model = MODELS.build(self.config.model)
        

    def forward(self, x):
        x = self.model.data_preprocessor(x, training=True)
        output = self.model.forward(x["inputs"],x["data_samples"],mode = "predict")
        data_samples = [d.to_dict() for d in output]
        pred = [d['pred_scores']['item'] for d in data_samples]
        y = [d['gt_labels']['item'] for d in data_samples]
        pred = torch.stack(pred)
        y = torch.tensor(y)
        return pred
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


        # print((batch['inputs']))
        # print("*"*20+" net "+"*"*20, "\n")
        # print(batch['data_samples'])
        
        # print("n_batch", len(loader), len(bx), len(by), type(by))
        
        # # for bx, by in tqdm(datamodule.train_dataloader()):
        # #     pass
        # # print("training data passed")

        # # for bx, by in tqdm(datamodule.val_dataloader()):
        # #     pass
        # # print("validation data passed")

        # for bx, by in tqdm(datamodule.test_dataloader()):
        #     print(bx + ' ' + by)
        #     pass
        # print("test data passed")

    @hydra.main(version_base="1.3", config_path=config_path, config_name="recog.yaml")
    def main(cfg: DictConfig):
        print(cfg)
        test_datamodule(cfg)

    main()
    