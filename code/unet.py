#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import itertools
from datetime import datetime

import lightning.pytorch as pl
import torch
import torchvision
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers.wandb import WandbLogger
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    Dice,
    MulticlassAccuracy,
    MulticlassF1Score,
    MultilabelCoverageError,
    MulticlassJaccardIndex,
)
from torchvision.transforms import InterpolationMode, Resize
import wandb

from dataset import BBBC039Segmentation


class UNet(LightningModule):
    """UNet DLA model lightning module implementation."""

    def __init__(
        self,
        learning_rate: float = 1e-4,
        in_channels: int = 3,
        out_channels: int = 3,
        init_features: int = 32,
        pretrained: bool = False,
        rand_augment: bool = False,
        rand_augment_n: int = None,
        rand_augment_m: float = None,
    ):
        super().__init__()

        # Set hyperparameters
        self.rand_augment = rand_augment
        self.rand_augment_n = rand_augment_n
        self.rand_augment_m = rand_augment_m
        if self.rand_augment:
            assert self.rand_augment_n is not None and self.rand_augment_m is not None
        self.learning_rate = learning_rate

        # Construct the model with the given parameters
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        self.net = torch.hub.load(
            "mateuszbuda/brain-segmentation-pytorch",
            "unet",
            in_channels=in_channels,
            out_channels=out_channels,
            init_features=init_features,
            pretrained=pretrained,
        )

        # Set the model loss function
        self.loss = torch.nn.functional.cross_entropy

        # Calculate the model number of hyperparameters
        self.num_parameters = sum(layer.numel() for layer in self.net.parameters())
        self.log("num_parameters", self.num_parameters, logger=True)

        # Define modular metrics
        self.train_f1 = MulticlassF1Score(num_classes=3)
        self.train_coverage = MultilabelCoverageError(num_labels=3)
        self.train_accuracy = MulticlassAccuracy(num_classes=3)
        self.train_dice = Dice(num_classes=3)
        self.val_f1 = MulticlassF1Score(num_classes=3)
        self.val_coverage = MultilabelCoverageError(num_labels=3)
        self.val_accuracy = MulticlassAccuracy(num_classes=3)
        self.val_dice = Dice(num_classes=3)
        self.test_f1 = MulticlassF1Score(num_classes=3)
        self.test_f1_binary = MulticlassF1Score(num_classes=3, ignore_index=2)
        self.test_coverage = MultilabelCoverageError(num_labels=3)
        self.test_coverage_binary = MultilabelCoverageError(
            num_labels=3, ignore_index=2
        )
        self.test_accuracy = MulticlassAccuracy(num_classes=3)
        self.test_accuracy_binary = MulticlassAccuracy(num_classes=3, ignore_index=2)
        self.test_dice = Dice(num_classes=3)
        self.test_dice_binary = Dice(num_classes=3, ignore_index=2)
        self.test_jid = MulticlassJaccardIndex(num_classes=3)
        self.test_jid_binary = MulticlassJaccardIndex(num_classes=3, ignore_index=2)

        # Shorthand for logger options
        self.train_log_opts = {"on_step": True, "on_epoch": True, "logger": True}
        self.val_log_opts = {"on_step": False, "on_epoch": True, "logger": True}
        self.test_log_opts = {"on_step": False, "on_epoch": True, "logger": True}

        # Save the hyperparameters for this model
        self.save_hyperparameters()

    def training_step(self, batch: tuple, batch_idx: int):
        # Load and inference
        images, labels = batch
        output = self.net(images)
        pred, labs = output.argmax(1), labels.int().argmax(1)
        loss = self.loss(output, labels)

        # Update metrics
        self.train_f1(pred, labs)
        self.train_coverage(output, labels)
        self.train_accuracy(pred, labs)
        self.train_dice(pred, labs)
        self.log("train_loss", loss, prog_bar=True, **self.train_log_opts)
        self.log("train_f1", self.train_f1, prog_bar=True, **self.train_log_opts)
        self.log("train_coverage", self.train_coverage, **self.train_log_opts)
        self.log("train_accuracy", self.train_accuracy, **self.train_log_opts)
        self.log("train_dice", self.train_dice, **self.train_log_opts)

        return loss

    def validation_step(self, batch: tuple, batch_idx: int):
        # Load and inference
        images, labels = batch
        output = self.net(images)
        pred, labs = output.argmax(1), labels.int().argmax(1)
        loss = self.loss(output, labels)

        # Update metrics
        self.val_f1(pred, labs)
        self.val_coverage(output, labels)
        self.val_accuracy(pred, labs)
        self.val_dice(pred, labs)
        self.log("val_loss", loss, **self.val_log_opts)
        self.log("val_f1", self.val_f1, prog_bar=True, **self.val_log_opts)
        self.log("val_coverage", self.val_coverage, **self.val_log_opts)
        self.log("val_accuracy", self.val_accuracy, **self.val_log_opts)
        self.log("val_dice", self.val_dice, **self.val_log_opts)
        return (output, labels)

    def on_validation_batch_end(self, output, batch: tuple, batch_idx: int):
        # Only log the first batch inference
        if batch_idx == 0:
            pred, labs = output
            self.logger.log_image("prediction", [wandb.Image(pred[0])])
            self.logger.log_image("true_label", [wandb.Image(labs[0])])

    def test_step(self, batch: tuple, batch_idx: int):
        # Load and inference
        images, labels = batch
        output = self.net(images)
        pred, labs = output.argmax(1), labels.int().argmax(1)
        loss = self.loss(output, labels)

        # Update metrics
        self.test_f1(pred, labs)
        self.test_coverage(output, labels)
        self.test_accuracy(pred, labs)
        self.test_dice(pred, labs)
        self.test_f1_binary(pred, labs)
        self.test_coverage_binary(output, labels)
        self.test_accuracy_binary(pred, labs)
        self.test_dice_binary(pred, labs)
        self.test_jid(pred, labs)
        self.test_jid_binary(pred, labs)
        self.log("test_loss", loss, **self.test_log_opts)
        self.log("test_f1", self.test_f1, **self.test_log_opts)
        self.log("test_f1_binary", self.test_f1_binary, **self.test_log_opts)
        self.log("test_coverage", self.test_coverage, **self.test_log_opts)
        self.log(
            "test_coverage_binary", self.test_coverage_binary, **self.test_log_opts
        )
        self.log("test_accuracy", self.test_accuracy, **self.test_log_opts)
        self.log(
            "test_accuracy_binary", self.test_accuracy_binary, **self.test_log_opts
        )
        self.log("test_dice", self.test_dice, **self.test_log_opts)
        self.log("test_dice_binary", self.test_dice_binary, **self.test_log_opts)
        self.log("test_jid", self.test_jid, **self.test_log_opts)
        self.log("test_jid_binary", self.test_jid_binary, **self.test_log_opts)

    def predict_step(self, batch: tuple, batch_idx: int):
        # Required for testing the model
        return self.net(batch[0]).argmax(1)

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        # Adam optimiser for model trainnig
        optimiser = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        return optimiser


#### Argparse for renewing logger

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=0, help="grid index")
args = parser.parse_args()

#### Define the training loop hyperparameter grid search

epochs = 10
batch_size = 8

baselines = {
    "rand_augment": [False],
    "rand_augment_n": [None],
    "rand_augment_m": [None],
    "init_features": [16, 32, 64],
    "learning_rate": [1e-3, 1e-4],
}

augmented = {
    "rand_augment": [True],
    "rand_augment_n": [2, 3, 4, 5],
    "rand_augment_m": [0.25, 0.5],
    "init_features": [16, 32, 64],
    "learning_rate": [1e-3, 1e-4],
}

baseline_grid = [
    dict(zip(baselines.keys(), x)) for x in itertools.product(*baselines.values())
]
augmented_grid = [
    dict(zip(augmented.keys(), x)) for x in itertools.product(*augmented.values())
]
parameter_set = (baseline_grid + augmented_grid)[args.n]


##### Define the main training loop


def train_loop(param_grid: dict, epochs: int, batch_size: int) -> None:
    """Hyperparameter training loop iteration"""

    # Setup data loaders
    transform = Resize((256, 256), InterpolationMode.NEAREST)  # For U-Net model
    train_dataset = BBBC039Segmentation(
        "./datasets/",
        subset="train",
        transform=transform,
        rand_augment=param_grid["rand_augment"],
        rand_augment_n=param_grid["rand_augment_n"],
        rand_augment_m=param_grid["rand_augment_m"],
    )
    val_dataset = BBBC039Segmentation(
        "./datasets/",
        subset="val",
        transform=transform,
        rand_augment=param_grid["rand_augment"],
        rand_augment_n=param_grid["rand_augment_n"],
        rand_augment_m=param_grid["rand_augment_m"],
    )
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)

    # Seed everything
    seed_everything(3419, workers=True)

    # Setup the run logger
    run_name = ""
    if param_grid["rand_augment"]:
        run_name += f"randaugment{param_grid['rand_augment_n']}_"
    else:
        run_name += "baseline_"
    run_name += "fastlr_" if param_grid["learning_rate"] == 1e-3 else ""
    run_name += f"unet_{'small_' if param_grid['init_features'] == 16 else ''}"
    run_name += f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    print(run_name)
    logger = WandbLogger(name=run_name, project="BBBC039-Segmentation", log_model="all")

    # Setup checkpoint and stopping callbacks
    checkpoint_cb = ModelCheckpoint(monitor="val_f1", mode="max")
    stopping_cb = EarlyStopping(
        monitor="val_f1",
        min_delta=0.00,
        patience=5,
        mode="max",
    )

    # Initialise the trainer
    trainer = Trainer(
        logger=logger,
        max_epochs=epochs,
        accelerator="auto",
        callbacks=[checkpoint_cb],
        default_root_dir="./logs",
        deterministic=True,
    )

    # Setup the lightning module and fit with the loaders
    unet = UNet(
        init_features=param_grid["init_features"],
        learning_rate=param_grid["learning_rate"],
        rand_augment=param_grid["rand_augment"],
        rand_augment_n=param_grid["rand_augment_n"],
        rand_augment_m=param_grid["rand_augment_m"],
    )
    trainer.fit(unet, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    train_loop(parameter_set, epochs, batch_size)
