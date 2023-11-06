#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import dataclasses
import itertools
from datetime import datetime
from typing import Callable

import wandb
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from torchvision.transforms import InterpolationMode, Resize

from bbbc039.modules import BBBC039DataModule, UNetDLAModule

SEED: int = 42
PROJECT_NAME: str = "BBBC039-Segmentation"
LOGGING_DIRECTORY: str = "./logs"
TRAIN_EPOCHS: int = 10
BATCH_SIZE: int = 8

# Parser for the dataset directory
parser = argparse.ArgumentParser()
parser.add_argument("--datapath", default="./datasets", help="dataset directory path")
args = parser.parse_args()

"""
Hyperparameter Grid Search
    Grid search ablations and baseline configuration is defined as a dataclass instance
    with a set of config values. Hyperparameter combinations are separated from baseline
    and ablations as the former requires specific values for the `rand_augment` fields.
"""


@dataclasses.dataclass
class Hyperparameters:
    init_features: int
    learning_rate: float
    rand_augment: bool
    rand_augment_n: int
    rand_augment_m: float


baseline_hparams = {
    "init_features": [16, 32, 64],
    "learning_rate": [1e-3, 1e-4],
    "rand_augment": [False],
    "rand_augment_n": [None],
    "rand_augment_m": [None],
}
ablation_hparams = {
    "init_features": baseline_hparams["init_features"],
    "learning_rate": baseline_hparams["learning_rate"],
    "rand_augment": [True],
    "rand_augment_n": [1, 3, 5],
    "rand_augment_m": [0.1, 0.25, 0.5],
}

hparams_grid = [
    dict(zip(baseline_hparams.keys(), x))
    for x in itertools.product(*baseline_hparams.values())
] + [
    dict(zip(ablation_hparams.keys(), x))
    for x in itertools.product(*ablation_hparams.values())
]
hparams_grid = [Hyperparameters(**v) for v in hparams_grid]


def run_model(root: str, hparams: Hyperparameters, epochs: int, batch_size: int):
    """
    Contained within a wandb experiment, perform an individual run by training the UNet-
    DLA model for the BBC039 dataset using the modified RandAugment implmentation for
    instance segmentation.

    Args:
        root (str): Root path to the BBBC039 dataset directory (from command line).
        hparams (Hyperparameters): Hyperparameter configuration values.
        epochs (int): Number of epochs for training the model.
        batch_size (int): Batch size for BBBC039 dataloaders.
    """
    # Set run name for wandb
    run_desc: str = "randaugment" if hparams.rand_augment else "baseline"
    ablation: str = run_desc
    if hparams.rand_augment:
        run_desc += "{:02d}".format(hparams.rand_augment_n)
    run_size: str = "l" if hparams.init_features == 64 else "s"
    run_size: str = "m" if hparams.init_features == 32 else run_size
    run_time: str = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    run_name: str = f"{run_desc}_{run_size}_unet_{run_time}"

    # Setup trainer logger
    logger_checkpoint: Callable = ModelCheckpoint(monitor="val_jaccard", mode="max")
    logger: WandbLogger = WandbLogger(
        project=PROJECT_NAME,
        name=run_name,
        log_model="all",
        save_dir=LOGGING_DIRECTORY,
    )

    # Setup datamodule and model
    resizing: list = [Resize((256, 256), interpolation=InterpolationMode.NEAREST)]
    bbbc039_datamodule: BBBC039DataModule = BBBC039DataModule(
        root,
        batch_size,
        resizing,
        hparams.rand_augment,
        hparams.rand_augment_n,
        hparams.rand_augment_m,
    )
    hparams_dict: dict = dataclasses.asdict(hparams)
    unet_model: UNetDLAModule = UNetDLAModule(**hparams_dict, ablation=ablation)

    # Initialise trainer
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="auto",
        deterministic=True,
        logger=logger,
        callbacks=[logger_checkpoint],
        default_root_dir=LOGGING_DIRECTORY,
    )

    # Performing fit and test
    seed_everything(SEED, workers=True)
    trainer.fit(unet_model, bbbc039_datamodule)
    trainer.test(unet_model, bbbc039_datamodule)
    wandb.finish()


if __name__ == "__main__":
    # Main runtime hyperparameter grid search
    for hparams in sorted(hparams_grid, reverse=True):
        # Completed runs are skipped; experiement is performed
        hparams_dict: dict = dataclasses.asdict(hparams)
        completed_runs: list = wandb.Api().runs(PROJECT_NAME)
        print(hparams)
        if any([hparams_dict == run.config for run in completed_runs]):
            print(f"Existing run found: {len(completed_runs)} completed")
            continue
        run_model(args.datapath, hparams, TRAIN_EPOCHS, BATCH_SIZE)
