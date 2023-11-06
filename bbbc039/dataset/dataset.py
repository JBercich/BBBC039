#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import zipfile
from enum import Enum, unique
from pathlib import Path
from typing import Callable

import numpy as np
import numpy.typing as npt
import requests
import skimage
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Normalize, ToTensor
from tqdm.auto import tqdm

from bbbc039.dataset.randaugment import RandAugmentSegmentation

MIN_NUCLEI_SIZE: int = 25
PIXEL_LEN_SPLIT: int = 232


class BBBC039(Dataset):
    """
    BBBC039 dataset for nuclei instance segmentation.

    Dataset Summary:
        The dataset was made using a Hoechst stain and fluorescence microscopy on high-
        throughput U2OS histopathological nuclear phenotypes from 200 different fields
        of view. Each image contains labelled nuceli with annotated overlapping regions
        for distinguishing nuclei boundaries within a 16-bit 520x696 frame. Metadata
        splits for training, validation and testing are provided.

    Dataset Source:
        This dataset was source from https://bbbc.broadinstitute.org/BBBC039. BBBC039v1
        Caicedo et al. 2018, available from the Broad Bioimage Benchmark Collection
        [Ljosa et al., Nature Methods, 2012].
    """

    DATASET_NAME: str = "bbbc039"
    IMAGES_EXT: str = ".tif"
    LABELS_EXT: str = ".png"
    SUBSETS: list[str] = ["train", "val", "test"]
    MAX_IMAGE_PIXEL: int = 4095

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

    def __init__(self, root: str, subset: str = "train", transform: list = None):
        super().__init__()

        # Validate parameters
        if subset not in self.SUBSETS:
            raise ValueError(f"subset '{self.subset}` invalid: select {self.SUBSETS}")

        # Setup instance attributes
        self.root: Path = Path(root).resolve()
        if self.root.name != self.DATASET_NAME:
            self.root: Path = self.root / self.DATASET_NAME
        self.root.mkdir(parents=True, exist_ok=True)
        self.subset, self.transform = subset, transform
        self.images_dir, self.labels_dir = self.root / "images", self.root / "labels"

        # Download and extract required dataset files
        mirror_directory: Path = self.root / "mirrors"
        mirror_directory.mkdir(exist_ok=True)
        for mirror in self.Mirror:
            filename, url = mirror.value
            filepath: Path = mirror_directory / filename
            if not filepath.exists():
                self._download(url, filepath)
            if not (self.root / filename).with_suffix("").exists():
                if "masks" in filename and self.labels_dir.exists():
                    continue
                self._extract(filepath, self.root)
                if "masks" in filename:
                    os.rename(self.root / "masks", self.labels_dir)

        # Extract dataset subset ids
        def _get_subset_ids(subset_filepath: Path) -> list[str]:
            with open(subset_filepath, "r") as subset_file:
                subset_ids = [i.split(".")[0] for i in subset_file.read().splitlines()]
            return subset_ids

        metadata_dir = self.root / "metadata"
        subset_ids_list = [
            ("train", _get_subset_ids(metadata_dir / "training.txt")),
            ("val", _get_subset_ids(metadata_dir / "validation.txt")),
            ("test", _get_subset_ids(metadata_dir / "test.txt")),
        ]

        # Create subset directories for images and labels
        for dirpath in [self.images_dir, self.labels_dir]:
            for subset_name, ids in subset_ids_list:
                directory = dirpath / subset_name
                directory.mkdir(exist_ok=True)
                for file in dirpath.iterdir():
                    if file.with_suffix("").name in ids:
                        file.rename(directory / file.name)

        # Collect preprocessing images and labels
        process_images, process_labels = [], []
        for subset_name in self.SUBSETS:
            for f in (self.images_dir / subset_name).iterdir():
                if any([f.with_suffix("").name in sset[1] for sset in subset_ids_list]):
                    process_images.append(f)
            for f in (self.labels_dir / subset_name).iterdir():
                if any([f.with_suffix("").name in sset[1] for sset in subset_ids_list]):
                    process_labels.append(f)
        process_images, process_labels = sorted(process_images), sorted(process_labels)
        process_queue = [i for i in zip(process_images, process_labels)]

        if len(process_queue):
            print(f"Preprocessing images and labels {self.root}")

        # Perform preprocessing operations
        idx = 1
        for image_path, label_path in process_queue:
            queue = zip(
                self._split_array(np.array(Image.open(image_path)), PIXEL_LEN_SPLIT),
                self._split_array(np.array(Image.open(label_path)), PIXEL_LEN_SPLIT),
            )
            for image, label in queue:
                filename = "{:08d}".format(idx)
                image = self._preprocess_image(image)
                label = self._preprocess_label(label)
                image_file = image_path.with_name(filename).with_suffix(self.IMAGES_EXT)
                label_file = label_path.with_name(filename).with_suffix(self.LABELS_EXT)
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
    def _extract(archive: Path, output_directory: Path):
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
    def _preprocess_image(image: npt.NDArray, mp: int = MAX_IMAGE_PIXEL) -> npt.NDArray:
        """
        Normalise images between 0-1 based on the dataset maximum pixel value, convert
        the image to 3-channel RGB by duplicate the grayscale pixel value and return as
        an 8-bit uint image array.

        Args:
            image (npt.NDArray): Image being preprocessed.
            mp (int, optional): Maximum pixel value. Defaults to MAX_IMAGE_PIXEL.

        Returns:
            npt.NDArray: Preprocessed image array.
        """
        return (np.stack((image / mp,) * 3, axis=-1) * 255).astype(np.uint8)

    @staticmethod
    def _preprocess_label(label: npt.NDArray, sz: int = MIN_NUCLEI_SIZE) -> npt.NDArray:
        """
        Remove micronuclei and added label channels, extract boundaries and create new
        label array with 3-channel background, nuclei and boundary classes.

        Args:
            label (npt.NDArray): Label to be preprocessed.
            sz (int, optional): Minimum size of nuclei. Defaults to MIN_NUCLEI_SIZE.

        Returns:
            npt.NDArray: Preprocessed nuclei.
        """
        # Remove micronuclei, label nuclei, remove excess channels
        label = skimage.morphology.label(label[:, :, 0])
        if label.max() > 1:
            label = skimage.morphology.remove_small_objects(label, min_size=sz)
        # Extract nuclei boundaries
        bounds = skimage.segmentation.find_boundaries(label)
        # Create new nuclei label mask
        new_label = np.zeros((label.shape + (3,)))
        new_label[(label == 0) & (bounds == 0), 0] = 1
        new_label[(label != 0) & (bounds == 0), 1] = 1
        new_label[bounds == 1, 2] = 1
        return new_label.astype(np.uint8)

    def __getitem__(self, idx: int) -> tuple:
        """
        Load image and label files into memory, perform any required pre-processing
        operations and given transforms to the images and labels.

        Args:
            idx (int): Index within the given subset collection.

        Returns:
            tuple: Loaded and transformed image and label tensors.
        """
        # Read in and perform required preparation transformations
        image, label = Image.open(self.image_ids[idx]), Image.open(self.label_ids[idx])
        image = Normalize((0.0601831,) * 3, (0.051409,) * 3)(ToTensor()(image))
        label = (ToTensor()(label) * 255).round()
        # Apply additional transformations
        if self.transform is not None:
            for transform in self.transform:
                if isinstance(transform, RandAugmentSegmentation):
                    image, label = transform(image, label)
                else:
                    image, label = transform(image), transform(label)
        return image, label

    def __len__(self):
        # Required dunder function for datasets
        return len(self.image_ids)

    def to_numpy(self):
        """
        Iterate and load each image and label (PNGs) into numpy arrays for the given
        subset of the dataset. The dataset will perform any transforms that are defined
        and convert each loaded image/label into a numpy array. Any resizing transforms
        will cause an error.

        Returns:
            tuple: Image and label numpy arrays, note that it must be the label PNGs.
        """
        loaded_images, loaded_labels = [], []
        for image, label in self:
            loaded_images.append(image.numpy()), loaded_labels.append(label.numpy())
        return np.array(loaded_images), np.array(loaded_labels)
