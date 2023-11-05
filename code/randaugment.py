#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import random

from torch import Tensor
from torchvision.transforms import (
    CenterCrop,
    Compose,
    RandomAffine,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)


class RandAugmentSegmentation:
    """Segmentation modified RandAugment implementation."""

    def __init__(self, n: int, m: float):
        self.n = n
        self.m = m

    def __call__(self, image: Tensor) -> Tensor:
        # Augmentations are defined with unique settings based on m
        augmentations = [
            RandomHorizontalFlip(self.m),
            RandomVerticalFlip(self.m),
            CenterCrop(200 * (1 - self.m)),
            RandomAffine(
                degrees=360 * self.m,
                shear=(self.m, self.m),
                translate=(self.m, self.m),
                scale=(1 - self.m, 1 - self.m),
            ),
        ]

        # Collect n random augmentation strategiesi
        selection = []
        for i in range(self.n):
            selection.append(random.choice(augmentations))

        # Perform all composed augmentations on the input image
        return Compose(selection)(image)
