#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import random
import shutil
import warnings
import zipfile
from enum import Enum, auto, unique
from pathlib import Path
from typing import Callable

import numpy as np
import numpy.typing as npt
import requests
import skimage
import torch
import torchvision
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import *
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm.auto import tqdm

from randaugment import RandAugmentSegmentation

warnings.filterwarnings("ignore", category=UserWarning)


class BBBC039Segmentation(Dataset):
    """BBBC039 dataset module."""

    dataset_name: str = "bbbc039"
    images_ext: str = ".tif"
    labels_ext: str = ".png"
    background_class: int = 0
    nucleus_class: int = 1
    boundary_class: int = 2
    min_nuclei_size: int = 25
    pixel_split: int = 232

    @unique
    class Mirror(tuple, Enum):
        """
        Mirrors are available from the BBC source organisation. This data is completely
        public but requires appropriate citations under any licence agreements defined
        by the BBC on its website where the mirrors are hosted. See the dataset source
        information for more details.
        """

        Images = (
            "images.zip",
            "https://data.broadinstitute.org/bbbc/BBBC039/images.zip",
        )
        Labels = (
            "masks.zip",
            "https://data.broadinstitute.org/bbbc/BBBC039/masks.zip",
        )
        Metadata = (
            "metadata.zip",
            "https://data.broadinstitute.org/bbbc/BBBC039/metadata.zip",
        )

    @unique
    class Subset(str, Enum):
        """
        Allowed BBBC039 segmentation subsets. These correspond to the available dataset
        metadata ID partitions and are publically available in the dataset metadata. For
        convenience, the different partitions are given in separate subset directories
        for both images and labels.
        """

        Train = "train"
        Val = "val"
        Test = "test"

    def __init__(
        self,
        root: str,
        subset: str = Subset.Train.value,
        force: bool = False,
        transform: Callable = None,
        rand_augment: bool = False,
        rand_augment_n: int = None,
        rand_augment_m: float = None,
    ):
        super().__init__()
        self.rand_augment = rand_augment
        self.rand_augment_n = rand_augment_n
        self.rand_augment_m = rand_augment_m
        if self.rand_augment:
            assert self.rand_augment_n is not None and self.rand_augment_m is not None

        # Validate input arguments
        if subset not in self.Subset._value2member_map_:
            raise ValueError("subset '{subset}` invalid: select [train, val, test]")
        self.transform = transform

        # Setup the root directory path
        self.root = Path(root).resolve()
        if self.root.name != self.dataset_name:
            self.root = self.root / self.dataset_name
        self.root.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.root / "images"
        self.labels_dir = self.root / "labels"

        # Download and extract required dataset files
        self.mirror_directory = self.root / "mirrors"
        self.mirror_directory.mkdir(exist_ok=True)
        for mirror in self.Mirror:
            filename, url = mirror.value
            filepath = self.mirror_directory / filename
            if not filepath.exists():
                self._download(url, filepath)
            if not (self.root / filename).with_suffix("").exists() or force:
                if "masks" in filename and self.labels_dir.exists() and not force:
                    continue
                self._extract(filepath, self.root)

        # Handling mask and label directory renaming
        masks_dir = self.root / "masks"
        if force and self.labels_dir.exists():
            shutil.rmtree(self.labels_dir)
        if not self.labels_dir.exists():
            os.rename(masks_dir, self.labels_dir)
        if masks_dir.exists():
            shutil.rmtree(masks_dir)

        # Extract dataset subset ids
        def _get_subset_ids(subset_filepath: Path) -> list[str]:
            with open(subset_filepath, "r") as subset_file:
                subset_ids = [i.split(".")[0] for i in subset_file.read().splitlines()]
            return subset_ids

        metadata_dir = self.root / "metadata"
        subsets = zip(
            self.Subset._value2member_map_.keys(),
            [
                _get_subset_ids(metadata_dir / "training.txt"),
                _get_subset_ids(metadata_dir / "validation.txt"),
                _get_subset_ids(metadata_dir / "test.txt"),
            ],
        )
        subsets = [i for i in subsets]

        # Create subset directories for images and labels
        for dirpath in [self.images_dir, self.labels_dir]:
            for subset_, ids in subsets:
                directory = dirpath / subset_
                directory.mkdir(exist_ok=True)
                for file in dirpath.iterdir():
                    if file.with_suffix("").name in ids:
                        file.rename(directory / file.name)

        # Collect preprocessing images and labels
        preprocess_images, preprocess_labels = [], []
        for subset_ in self.Subset:
            for file in (self.images_dir / subset_.value).iterdir():
                if any([file.with_suffix("").name in sset[1] for sset in subsets]):
                    preprocess_images.append(file)
            for file in (self.labels_dir / subset_.value).iterdir():
                if any([file.with_suffix("").name in sset[1] for sset in subsets]):
                    preprocess_labels.append(file)
        preprocess_images = sorted(preprocess_images)
        preprocess_labels = sorted(preprocess_labels)
        preprocess_queue = zip(preprocess_images, preprocess_labels)

        if len(preprocess_images):
            print(f"Preprocessing images and labels {self.root}")

        # Perform preprocessing operations
        idx = 1
        for image_path, label_path in preprocess_queue:
            process_queue = zip(
                self._split_array(np.array(Image.open(image_path)), self.pixel_split),
                self._split_array(np.array(Image.open(label_path)), self.pixel_split),
            )
            for image, label in process_queue:
                filename = "{:08d}".format(idx)
                image = self._preprocess_image(image)
                label = self._preprocess_label(label, self.min_nuclei_size)
                image_file = image_path.with_name(filename).with_suffix(self.images_ext)
                label_file = label_path.with_name(filename).with_suffix(self.labels_ext)
                Image.fromarray(image).save(image_file)
                Image.fromarray(label).save(label_file)
                idx += 1
            image_path.unlink(), label_path.unlink()

        # Setup data image and label ids
        self.image_ids = [p for p in (self.images_dir / subset).iterdir()]
        self.label_ids = [p for p in (self.labels_dir / subset).iterdir()]
        self.image_ids = sorted(self.image_ids)
        self.label_ids = sorted(self.label_ids)

    @staticmethod
    def _download(url: str, output_path: Path):
        """
        Download a file from a URL into an output path. A progress bar is created and is
        updated while the dataset is being streamed into its file location. This will
        overwrite any file that it writes to.

        Args:
            url (str): URL from which the file is downloaded.
            output_path (Path): PosixPath to the download output file location.
        """
        # Download a file from a url to the output path
        response = requests.get(url, stream=True)
        response_size = int(response.headers["Content-Length"])
        # Define a progress bar for downloading the file
        desc = f"Downloading {url}"
        pbar = tqdm(desc=desc, total=response_size, unit_scale=True, unit="B")
        # Write the downloaded chunks to download_path
        with open(output_path, "wb") as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
                pbar.update(1024)

    @staticmethod
    def _extract(archive: Path, output_directory: Path) -> None:
        """
        Extract an archive from a Posix Path into an output directory. Any extracted
        archive file that is given will overwrite any existing archive contents in the
        output directory.

        Args:
            archive (Path): Posix path to am archive file that is to be extracted.
            output_directory (Path): Posix path to a directory where archive zipfiles
                are extracted into.
        """
        print("Extracting", archive)
        with zipfile.ZipFile(archive, "r") as archive:
            archive.extractall(output_directory)

    @staticmethod
    def _split_array(array: npt.NDArray, npixels: int):
        """
        Utility function used to split a numpy array into a set of square segments. The
        array is spliced into the desired range and then `nrows x ncols` sub-arrays are
        generated and yielded based on a given number of square pixel dimensions.

        Args:
            array (npt.NDArray): Numpy array that is being split into equivalent square
                segments with the given dimensions.
            npixels (int): Number of pixels for the side length of segmented squares.

        Yields:
            npt.NDArray: Unique square segment generated from the given array.
        """
        nrows, ncols = array.shape[0] // npixels, array.shape[1] // npixels
        for rblock in np.hsplit(array[: npixels * nrows, : npixels * ncols], ncols):
            for cblock in np.vsplit(rblock, nrows):
                yield cblock

    @staticmethod
    def _preprocess_image(image: npt.NDArray) -> npt.NDArray:
        # Normalise between 0 and 1
        image = image / 4095
        # Convert to 3-channel uint RGB images
        image = (np.stack((image,) * 3, axis=-1) * 255).astype(np.uint8)
        return image

    @staticmethod
    def _preprocess_label(label: npt.NDArray, min_nuclei_size: int) -> npt.NDArray:
        # Remove micronuclei, label nuclei, remove excess channels
        label = skimage.morphology.remove_small_objects(
            skimage.morphology.label(label[:, :, 0]),
            min_size=min_nuclei_size,
        )
        # Extract nuclei boundaries
        bounds = skimage.segmentation.find_boundaries(label)
        # Create new nuclei label mask
        new_label = np.zeros((label.shape + (3,)))
        new_label[(label == 0) & (bounds == 0), 0] = 1
        new_label[(label != 0) & (bounds == 0), 1] = 1
        new_label[bounds == 1, 2] = 1
        return new_label.astype(np.uint8)

    def __getitem__(self, idx: int):
        # Read in and perform the required preprocessing actions
        image, label = Image.open(self.image_ids[idx]), Image.open(self.label_ids[idx])
        image = Normalize((0.0601831,) * 3, (0.051409,) * 3)(ToTensor()(image))
        label = (ToTensor()(label) * 255).round()
        # Apply transformations and coerce image and mask into tensors
        if self.rand_augment:
            augment = RandAugmentSegmentation(self.rand_augment_n, self.rand_augment_m)
            joined = torch.cat((image.unsqueeze(0), label.unsqueeze(0)), 0)
            transformed_joined = augment(joined)
            image = transformed_joined[0]
            label = transformed_joined[1]
        image = self.transform(image) if self.transform else image
        label = self.transform(label) if self.transform else label
        return image, label

    def __len__(self):
        return len(self.image_ids)

    def to_numpy(self):
        """
        Iterate and load each image and label (PNGs) into numpy arrays for the given
        subset of the dataset. The dataset will perform any transforms that are defined
        and convert each loaded image/label into a numpy array.

        Returns:
            tuple: Image and label numpy arrays, note that it must be the label PNGs.
        """
        loaded_images = []
        loaded_labels = []
        for image, label in self:
            # Load and transform the image and masks to numpy arrays
            loaded_images.append(image.numpy())
            loaded_labels.append(label.numpy())
        # Convert the final lists to arrays
        return np.array(loaded_images), np.array(loaded_labels)
