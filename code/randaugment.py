#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import random
from abc import ABC, abstractmethod
from typing import Callable

from torch import Tensor
from torchvision.transforms import (
    CenterCrop,
    Compose,
    RandomAffine,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)


class RandAugmentSegmentation:
    """
    Modified RandAugment for segmentation models.

    Considering the need for tuning individual augmentation operations for the magnitude
    parameter `m`, hyperparameters `n,m` are left consistent for single model instances
    and tuned as a static value rather than specifically for each operation. Separate
    operations are defined as subclasses take as input the hyperparameter `m`.
    """

    class Operation(ABC):
        @abstractmethod
        def __init__(self, m: float, image: Tensor) -> Callable:
            return None

    class HorizontalFlip(Operation):
        def __init__(self, m: float, image: Tensor) -> Callable:
            return RandomHorizontalFlip(m)

    class VerticalFlip(Operation):
        def __init__(self, m: float, image: Tensor) -> Callable:
            return RandomVerticalFlip(m)

    class CentreCrop(Operation):
        def __init__(self, m: float, image: Tensor) -> Callable:
            return CenterCrop((image.shape[-2] * (1 - m), image.shape[-1] * (1 - m)))

    class Translate(Operation):
        def __init__(self, m: float, image: Tensor) -> Callable:
            return RandomAffine(0, translate=(m, m))

    class ShearX(Operation):
        def __init__(self, m: float, image: Tensor) -> Callable:
            return RandomAffine(0, shear=(180 * m, 180 * m, 0, 0))

    class ShearY(Operation):
        def __init__(self, m: float, image: Tensor) -> Callable:
            return RandomAffine(0, shear=(0, 0, 180 * m, 180 * m))

    class RotateR(Operation):
        def __init__(self, m: float, image: Tensor) -> Callable:
            return RandomAffine((0, 360 * m))

    class RotateL(Operation):
        def __init__(self, m: float, image: Tensor) -> Callable:
            return RandomAffine((360 * m, 0))

    class Rescale(Operation):
        def __init__(self, m: float, image: Tensor) -> Callable:
            return RandomAffine(0, scale=(1 - m, 1))

    OPERATIONS: list[Operation] = [
        HorizontalFlip,
        VerticalFlip,
        CentreCrop,
        Translate,
        ShearX,
        ShearY,
        RotateR,
        RotateL,
        Rescale,
    ]

    def __init__(self, n: int, m: float):
        self.n = n
        self.m = m

    def __call__(self, image: Tensor) -> Tensor:
        random_operations: list[self.Operation] = []
        for _ in range(self.n):
            operation: Callable = random.choice(self.OPERATIONS)(self.m, image)
            random_operations.append(operation)
        return Compose(random_operations)(image)
