#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import wandb
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers import WandbLogger
from torchmetrics.classification import (Dice, MulticlassAccuracy,
                                         MulticlassF1Score,
                                         MulticlassJaccardIndex)

NUM_CLASSES: int = 3


class UNetDLAModule(LightningModule):
    """UNet-DLA Lightning Module."""

    TORCH_HUB_SOURCE: str = "mateuszbuda/brain-segmentation-pytorch"
    TORCH_HUB_NAME: str = "unet"

    def __init__(
        self,
        pretrained: bool = False,
        in_channels: int = 3,
        out_channels: int = 3,
        init_features: int = 32,
        learning_rate: float = 1e-4,
        **additional_parameters,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_features = init_features
        self.pretrained = pretrained

        # Initialise model
        self.net = torch.hub.load(
            self.TORCH_HUB_SOURCE,
            self.TORCH_HUB_NAME,
            pretrained=self.pretrained,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            init_features=self.init_features,
        )

        # Set the model loss function
        self.loss = torch.nn.functional.cross_entropy

        # Train metrics
        self.train_dice = Dice(NUM_CLASSES)
        self.train_f1score = MulticlassF1Score(NUM_CLASSES)
        self.train_jaccard = MulticlassJaccardIndex(NUM_CLASSES)
        self.train_accuracy = MulticlassAccuracy(NUM_CLASSES)
        # Validation metrics
        self.val_dice = Dice(NUM_CLASSES)
        self.val_f1score = MulticlassF1Score(NUM_CLASSES)
        self.val_jaccard = MulticlassJaccardIndex(NUM_CLASSES)
        self.val_accuracy = MulticlassAccuracy(NUM_CLASSES)
        # Test metrics
        self.test_dice = Dice(NUM_CLASSES)
        self.test_f1score = MulticlassF1Score(NUM_CLASSES)
        self.test_jaccard = MulticlassJaccardIndex(NUM_CLASSES)
        self.test_accuracy = MulticlassAccuracy(NUM_CLASSES)
        self.test_dice_bin = Dice(NUM_CLASSES, ignore_index=2)
        self.test_f1score_bin = MulticlassF1Score(NUM_CLASSES, ignore_index=2)
        self.test_jaccard_bin = MulticlassJaccardIndex(NUM_CLASSES, ignore_index=2)
        self.test_accuracy_bin = MulticlassAccuracy(NUM_CLASSES, ignore_index=2)
        self.logger_opts = {"on_epoch": True, "logger": True}

        self.save_hyperparameters()

    def training_step(self, batch: tuple, batch_idx: int):
        # Load and inference
        images, labels = batch
        output = self.net(images)
        losses = self.loss(output, labels)
        pred = output.argmax(1).int()
        labs = labels.int().argmax(1)
        # Update metrics
        self.train_dice(pred, labs)
        self.train_f1score(pred, labs)
        self.train_jaccard(pred, labs)
        self.train_accuracy(pred, labs)
        self.log("train_dice", self.train_dice, **self.logger_opts)
        self.log("train_f1score", self.train_f1score, prog_bar=True, **self.logger_opts)
        self.log("train_jaccard", self.train_jaccard, **self.logger_opts)
        self.log("train_accuracy", self.train_accuracy, **self.logger_opts)
        self.log("train_loss", losses, prog_bar=True, **self.logger_opts)
        return losses

    def validation_step(self, batch: tuple, batch_idx: int):
        # Load and inference
        images, labels = batch
        output = self.net(images)
        losses = self.loss(output, labels)
        pred = output.argmax(1).int()
        labs = labels.int().argmax(1)
        # Update metrics
        self.val_dice(pred, labs)
        self.val_f1score(pred, labs)
        self.val_jaccard(pred, labs)
        self.val_accuracy(pred, labs)
        self.log("val_dice", self.val_dice, **self.logger_opts)
        self.log("val_f1score", self.val_f1score, prog_bar=True, **self.logger_opts)
        self.log("val_jaccard", self.val_jaccard, **self.logger_opts)
        self.log("val_accuracy", self.val_accuracy, **self.logger_opts)
        self.log("val_loss", losses, **self.logger_opts)
        return (output, labels)

    def on_validation_batch_end(self, output: tuple, batch: tuple, batch_idx: int):
        # Log first batch inference
        if batch_idx == 0:
            pred, labs = output
            pred = torch.nn.functional.softmax(pred, dim=1)
            if isinstance(self.logger, WandbLogger):
                self.logger.log_image("input", [wandb.Image(batch[0])])
                self.logger.log_image("prediction", [wandb.Image(pred[0])])
                self.logger.log_image("true_label", [wandb.Image(labs[0])])

    def test_step(self, batch: tuple, batch_idx: int):
        # Load and inference
        images, labels = batch
        output = self.net(images)
        losses = self.loss(output, labels)
        pred = output.argmax(1).int()
        labs = labels.int().argmax(1)
        # Update metrics
        self.test_dice(pred, labs)
        self.test_f1score(pred, labs)
        self.test_jaccard(pred, labs)
        self.test_accuracy(pred, labs)
        self.log("test_dice", self.test_dice, **self.logger_opts)
        self.log("test_f1score", self.test_f1score, **self.logger_opts)
        self.log("test_jaccard", self.test_jaccard, **self.logger_opts)
        self.log("test_accuracy", self.test_accuracy, **self.logger_opts)
        self.log("test_loss", losses, **self.logger_opts)
        self.test_dice_bin(pred, labs)
        self.test_f1score_bin(pred, labs)
        self.test_jaccard_bin(pred, labs)
        self.test_accuracy_bin(pred, labs)
        self.log("test_dice_bin", self.test_dice_bin, **self.logger_opts)
        self.log("test_f1score_bin", self.test_f1score_bin, **self.logger_opts)
        self.log("test_jaccard_bin", self.test_jaccard_bin, **self.logger_opts)
        self.log("test_accuracy_bin", self.test_accuracy_bin, **self.logger_opts)

    def predict_step(self, batch: tuple, batch_idx: int):
        return self.net(batch[0]).argmax(1)

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        # Adam optimiser for model trainnig
        optimiser = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        return optimiser
