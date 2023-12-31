from typing import Any

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics import Accuracy, F1Score, Precision, Recall


class TokenClassificationModule(LightningModule):
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
        num_classes: int
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        metric_prams = dict(
            task="multiclass", num_classes=self.hparams.num_classes, ignore_index=-100
        )

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(**metric_prams)
        self.val_acc = Accuracy(**metric_prams)
        self.test_acc = Accuracy(**metric_prams)

        self.train_f1 = F1Score(**metric_prams,
                                average='macro')
        self.val_f1 = F1Score(**metric_prams,
                              average='macro')
        self.test_f1 = F1Score(**metric_prams,
                               average='macro')

        self.train_precision = Precision(**metric_prams, average='macro')
        self.val_precision = Precision(**metric_prams, average='macro')
        self.test_precision = Precision(**metric_prams, average='macro')

        self.train_recall = Recall(**metric_prams, average='macro')
        self.val_recall = Recall(**metric_prams, average='macro')
        self.test_recall = Recall(**metric_prams, average='macro')

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(self, batch: Any):
        out = self.forward(batch)
        loss = out.loss
        preds = out.logits.argmax(axis=-1)
        y = batch['labels']
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.train_f1(preds, targets)
        self.train_precision(preds, targets)
        self.train_recall(preds, targets)

        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/precession", self.train_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/recall", self.train_recall, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_f1(preds, targets)
        self.val_precision(preds, targets)
        self.val_recall(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/precession", self.val_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recall", self.val_recall, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        acc = self.val_f1.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/f1_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        loss, preds, targets = self.model_step(batch)

        return {
            'prediction' : preds.detach().cpu(),
            'targets': targets.detach().cpu(),
        }

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_f1(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)

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
                    "monitor": "val/f1",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = TokenClassificationModule(None, None, None)
