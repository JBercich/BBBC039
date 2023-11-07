#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from bbbc039.dataset import BBBC039, RandAugmentSegmentation


class BBBC039DataModule(LightningDataModule):
    """BBBC039 Lightning DataModule."""

    def __init__(
        self,
        root: str,
        batch_size: int = 8,
        transform: list = [],
        rand_augment: bool = False,
        rand_augment_n: int = None,
        rand_augment_m: float = None,
    ):
        super().__init__()
        self.root: str = root
        self.transform: list = transform
        self.batch_size: int = batch_size
        self.rand_augment: bool = rand_augment
        self.rand_augment_n: int = rand_augment_n
        self.rand_augment_m: float = rand_augment_m
        if rand_augment:
            assert rand_augment_n is not None and rand_augment_m is not None
            transform.append(RandAugmentSegmentation(rand_augment_n, rand_augment_m))
        self.save_hyperparameters(ignore="transform")

    def prepare_data(self):
        # Automatically download and prepare dataset in root directory
        BBBC039(self.root, transform=self.transform)

    def setup(self, stage: str):
        # Create different datasets for separate subsets
        if stage == "fit":
            self.train_data = BBBC039(self.root, transform=self.transform)
            self.val_data = BBBC039(self.root, transform=self.transform, subset="val")
        if stage == "test":
            self.test_data = BBBC039(self.root, transform=self.transform, subset="test")
        if stage == "predict":
            self.test_data = BBBC039(self.root, transform=self.transform, subset="test")

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)
