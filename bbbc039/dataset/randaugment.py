#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import random
from abc import ABC, abstractstaticmethod
from typing import Callable

from torch import Tensor, cat
from torchvision.transforms import (
    CenterCrop,
    ColorJitter,
    Compose,
    RandomAffine,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    functional,
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
        @abstractstaticmethod
        def build(m: float, image: Tensor) -> Callable:
            return None

    class HorizontalFlip(Operation):
        @staticmethod
        def build(m: float, image: Tensor) -> Callable:
            return RandomHorizontalFlip(m)

    class VerticalFlip(Operation):
        @staticmethod
        def build(m: float, image: Tensor) -> Callable:
            return RandomVerticalFlip(m)

    class CentreCrop(Operation):
        @staticmethod
        def build(m: float, image: Tensor) -> Callable:
            h, w = image.shape[-2:]
            phi = random.randint(90, 100) / 100
            return CenterCrop((int(h * (1 - m) * phi), int(w * (1 - m) * phi)))

    class Translate(Operation):
        @staticmethod
        def build(m: float, image: Tensor) -> Callable:
            return RandomAffine(0, translate=(m, m))

    class ShearX(Operation):
        @staticmethod
        def build(m: float, image: Tensor) -> Callable:
            return RandomAffine(
                0, shear=(random.randint(0, int(90 * m)), int(90 * m), 0, 0)
            )

    class ShearY(Operation):
        @staticmethod
        def build(m: float, image: Tensor) -> Callable:
            return RandomAffine(
                0, shear=(0, 0, random.randint(0, int(90 * m)), int(90 * m))
            )

    class Rescale(Operation):
        @staticmethod
        def build(m: float, image: Tensor) -> Callable:
            return RandomAffine(0, scale=(1 - m, 1))

    class Rotate(Operation):
        @staticmethod
        def build(m: float, image: Tensor) -> Callable:
            return RandomAffine((random.randint(0, int(360 * m)), int(360 * m)))

    class Brightness(Operation):
        @staticmethod
        def build(m: float, image: Tensor) -> Callable:
            phi: int = 10
            return ColorJitter(brightness=m)

    class Saturation(Operation):
        @staticmethod
        def build(m: float, image: Tensor) -> Callable:
            return ColorJitter(saturation=m)

    class Contrast(Operation):
        @staticmethod
        def build(m: float, image: Tensor) -> Callable:
            return ColorJitter(contrast=m)

    class Solarize(Operation):
        @staticmethod
        def build(m: float, image: Tensor) -> Callable:
            return lambda img: functional.solarize(
                img, threshold=max(random.random(), m)
            )

    class Hue(Operation):
        @staticmethod
        def build(m: float, image: Tensor) -> Callable:
            return ColorJitter(hue=min(0.5, m * random.random()))

    OPERATIONS: list[Operation] = [
        HorizontalFlip,
        VerticalFlip,
        CentreCrop,
        Translate,
        ShearX,
        ShearY,
        Rescale,
        Rotate,
    ]

    IMAGE_OPERATIONS: list[Operation] = [
        Brightness,
        Saturation,
        Contrast,
        Solarize,
        Hue,
    ]

    def __init__(self, n: int, m: float):
        self.n = n
        self.m = m

    def __call__(self, image: Tensor = None, label: Tensor = None) -> Tensor:
        # Collect random operations for both tensors and just for image
        operation_count = self.n // 2
        random_operations: list[self.Operation] = []
        for _ in range(operation_count):
            op: Callable = random.choice(self.OPERATIONS).build(self.m, image)
            random_operations.append(op)
        random_image_operations: list[self.Operation] = []
        for _ in range(self.n - operation_count):
            op: Callable = random.choice(self.IMAGE_OPERATIONS).build(self.m, image)
            random_image_operations.append(op)

        # Apply operation depending on which tensors are provided
        if image is not None and label is not None:
            combined = cat((image.unsqueeze(0), label.unsqueeze(0)), 0)
            return Compose(random_operations)(combined)
        elif label is not None:
            return Compose(random_operations)(label)
        elif image is not None:
            image = Compose(random_operations)(image)
        else:
            raise ValueError("at least one image or label must be augmented")
        return Compose(random_image_operations)(image)
