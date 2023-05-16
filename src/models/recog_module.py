from typing import Any
from mmengine import Config
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

class RegLitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        # config_file : str = 'mmaction2/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py',
        num_classes : int = 2
    ): 
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # self.config = Config.fromfile(config_file)
        # self.model = MODELS.build(self.config.model)
        # self.model.cls_head.num_classes = num_classes
        self.net = net
        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes = num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes = num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(self, batch: Any):
        param = (next(self.parameters()))
        device = param.device
        if (batch['inputs'][0].device != device):
            for key, value in batch.items():
                for i, input in enumerate(value):
                    # if torch.is_tensor(input):
                    batch[key][i] = input.to(device)
        y = batch['data_samples']
        data_samples = [d.to_dict() for d in y]
        y = [d['gt_labels']['item'] for d in data_samples]
        y = torch.tensor(y)
        logits = self.forward(batch)
        if (y.device != device):
            y=y.to(device)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        # y = batch['data_samples']
        # data_samples = [d.to_dict() for d in y]
        # y = [d['gt_labels']['item'] for d in data_samples]
        # y = torch.tensor(y)
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        y = batch['data_samples']
        data_samples = [d.to_dict() for d in y]
        y = [d['gt_labels']['item'] for d in data_samples]
        y = torch.tensor(y)

        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    # read config file from configs/model/dlib_resnet.yaml
    import pyrootutils
    from omegaconf import DictConfig
    import hydra

    # find paths
    path = pyrootutils.find_root(
        search_from=__file__, indicator=".project-root")
    config_path = str(path / "configs" / "model")
    output_path = path / "outputs"
    print("paths", path, config_path, output_path)

    # def test_net(cfg):
    #     net = hydra.utils.instantiate(cfg.net)
    #     print("*"*20+" net "+"*"*20, "\n", net)
    #     output = net(torch.randn(2,2,3,2,224,224))
    #     print("output", output.shape)

    def test_module(cfg):
        datamodule = hydra.utils.instantiate(cfg.data)
        datamodule.prepare_data()
        datamodule.setup()
        loader = datamodule.train_dataloader()
        # # loader
        batch = next(iter(loader))
        module = hydra.utils.instantiate(cfg.data)
        output = module(torch.randn(2,2,3,2,224,224))
        # print("module output", output.shape)

    @hydra.main(version_base="1.3", config_path=config_path, config_name="recog.yaml")
    def main(cfg: DictConfig):
        print(cfg)
        # test_net(cfg)
        test_module(cfg)

    main()
