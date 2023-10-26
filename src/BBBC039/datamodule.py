#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from typing import Callable, Final

import pytorch_lightning as pl
import torchvision as tv
from torch.utils.data import DataLoader

from .dataset import BBBC039


class BBBC039DataModule(pl.LightningDataModule):
    """
    BBBC039 data module implementation.

    Default normalisation transform is given as a default transformation, but if added
    transforms are applied, the normalisation should be applied accordingly post image
    callables.
    """

    normalisation_transform: Final[Callable] = tv.transforms.Compose(
        [
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=(262.3408031194739), std=(220.18462229587527)),
        ]
    )

    def __init__(
        self,
        root: str = "data",
        batch_size: int = 4,
        imgs_transform: Callable = normalisation_transform,
    ):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.imgs_transform = imgs_transform

    def prepare_data(self):
        # Download and preprocess
        BBBC039(self.root, subset=None, download=True)

    def setup(self, stage: str):
        if stage == "fit":
            # Model fitting with train and validation sets
            self.bbc039_train = BBBC039(self.root, "training", self.imgs_transform)
            self.bbc039_val = BBBC039(self.root, "validation", self.imgs_transform)
        elif stage == "test":
            # Model testing with the test set
            self.bbc039_test = BBBC039(self.root, "test", self.imgs_transform)
        elif stage == "predict":
            # Model prediction uses the same test set
            self.bbc039_predict = BBBC039(self.root, "test", self.imgs_transform)

        def train_dataloader(self):
            return DataLoader(self.bbc039_train, batch_size=self.batch_size)

        def val_dataloader(self):
            return DataLoader(self.bbc039_val, batch_size=self.batch_size)

        def test_dataloader(self):
            return DataLoader(self.bbc039_test, batch_size=self.batch_size)

        def predict_dataloader(self):
            return DataLoader(self.bbc039_predict, batch_size=self.batch_size)
