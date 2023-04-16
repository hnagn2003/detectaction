from mmaction.utils import register_all_modules

register_all_modules(init_default_scope=True)
from typing import Any, Dict, Optional, Tuple
import os.path as osp
import torch
from lightning import LightningDataModule
from mmengine.runner import Runner

import src.utils.recog.datamodule 

class RegDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.
    A DataModule implements 6 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test
    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.
    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 2,
        num_workers: int = 0,
        pin_memory: bool = False,
        ann_file_train: str = "kinetics_tiny_train_video.txt",
        data_root_train: str = "data/kinetics400_tiny/",
        ann_file_val: str = "kinetics_tiny_val_video.txt",
        data_root_val: str = "data/kinetics400_tiny/",
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.train_pipeline_cfg = [
            dict(type="VideoInit"),
            dict(type="VideoSample", clip_len=16, num_clips=1, test_mode=False),
            dict(type="VideoDecode"),
            dict(type="VideoResize", r_size=256),
            dict(type="VideoCrop", c_size=224),
            dict(type="VideoFormat"),
            dict(type="VideoPack"),
        ]
        self.val_pipeline_cfg = [
            dict(type="VideoInit"),
            dict(type="VideoSample", clip_len=16, num_clips=5, test_mode=True),
            dict(type="VideoDecode"),
            dict(type="VideoResize", r_size=256),
            dict(type="VideoCrop", c_size=224),
            dict(type="VideoFormat"),
            dict(type="VideoPack"),
        ]
        self.train_dataset_cfg = dict(
            type="DatasetZelda",
            ann_file=ann_file_train,
            pipeline=self.train_pipeline_cfg,
            data_root=data_root_train,
            data_prefix=dict(video="train"),
        )

        self.val_dataset_cfg = dict(
            type="DatasetZelda",
            ann_file=ann_file_val,
            pipeline=self.val_pipeline_cfg,
            data_root=data_root_val,
            data_prefix=dict(video="val"),
        )

        self.train_dataloader_cfg = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=False,
            sampler=dict(type="DefaultSampler", shuffle=True),
            dataset=self.train_dataset_cfg,
        )

        self.val_dataloader_cfg = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=False,
            sampler=dict(type="DefaultSampler", shuffle=False),
            dataset=self.val_dataset_cfg,
        )
        
        
        # self.transforms = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        # )

        # self.data_train: Optional[Dataset] = None
        # self.data_val: Optional[Dataset] = None
        # self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        return 10

    def prepare_data(self):
        """Download data if needed.
        Do not use it to assign state (self.x = y).
        """
        pass
        # MNIST(self.hparams.data_dir, train=True, download=True)
        # MNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        # if not self.data_train and not self.data_val and not self.data_test:
        #     trainset = MNIST(
        #         self.hparams.data_dir, train=True, transform=self.transforms
        #     )
        #     testset = MNIST(
        #         self.hparams.data_dir, train=False, transform=self.transforms
        #     )
        #     dataset = ConcatDataset(datasets=[trainset, testset])
        #     self.data_train, self.data_val, self.data_test = random_split(
        #         dataset=dataset,
        #         lengths=self.hparams.train_val_test_split,
        #         generator=torch.Generator().manual_seed(42),
        #     )

    def train_dataloader(self):
        return Runner.build_dataloader(dataloader=self.train_dataloader_cfg)

    def val_dataloader(self):
        return Runner.build_dataloader(dataloader=self.val_dataloader_cfg)

    def test_dataloader(self):
        return Runner.build_dataloader(dataloader=self.val_dataloader_cfg)

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import pyrootutils
    from omegaconf import DictConfig
    import hydra
    import numpy as np
    from PIL import Image, ImageDraw
    from tqdm import tqdm
    # datamodule: LightningDataModule = RegDataModule()
    # print(datamodule)
    # datamodule.prepare_data()
    # datamodule.setup()
    # loader = datamodule.train_dataloader()
    # # loader
    # batch = next(iter(loader))
    # print(batch)


    path = pyrootutils.find_root(
        search_from=__file__, indicator=".project-root")
    config_path = str(path / "configs" / "data")
    output_path = path / "outputs"
    print("root", path, config_path)
    # pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


    def test_datamodule(cfg: DictConfig):
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg)
        print(datamodule)
        datamodule.prepare_data()
        datamodule.setup()
        loader = datamodule.train_dataloader()
        # # loader
        batch = next(iter(loader))
        print(batch)
        
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