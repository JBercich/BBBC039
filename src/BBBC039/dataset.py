#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import hashlib
import math
import zipfile
from enum import Enum
from pathlib import Path
from typing import Callable, Final

import numpy as np
import requests
import skimage
import torch
import torchvision as tv
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class BBBC039(Dataset):
    """
    BBBC039 dataset.

    Segmentation mask dataset for U2OS cell nuclei obtained through fluorescence
    microscopy. Contains approximately 23,000 individually manually annotated nuclei.
    TIF image and PNG mask files are 520x696 pixels.

    Sourced from: https://bbbc.broadinstitute.org/BBBC039
        Caicedo et al. 2018, available from the Broad Bioimage Benchmark Collection
        [Ljosa et al., Nature Methods, 2012]

    Dataset has resource checksums, re-download dataset with download=True force=True if
    this is an issues for an existing dataset (removes cache files or other issues). Any
    preprocessing is also performed at runtime if not yet complete.
    """

    # Dataset resource hooks and structure
    resources_subsets: Final[list[str]] = ["training", "test", "validation"]
    resources_hook: Final[str] = "https://data.broadinstitute.org/bbbc/BBBC039/"
    resources: Final[Enum] = Enum(
        "resource",
        {
            "images": (
                resources_hook + "images.zip",
                "71d17e964654c093922f997e851be844",
            ),
            "masks": (
                resources_hook + "masks.zip",
                "a43f7ea4843bc8691f82f586aab7c70c",
            ),
            "metadata": (
                resources_hook + "metadata.zip",
                "75c9af146be5bb7eaa0c3aae6bf9b32f",
            ),
        },
    )

    # Preprocessing masks dataset
    preprocessed_checksum: Final[str] = "ae8d67357a747bb8cd57d86adec6d238"
    preprocessed_dirname: Final[str] = "masks_"

    # File extensions
    images_extension: Final[str] = ".tif"
    masks_extension: Final[str] = ".png"

    # Downloading chunk settings
    download_chunksize: Final[int] = 1024

    # Mask preprocessing settings for image manipulation
    min_nuclei_size: Final[int] = 25
    nuclei_border_w: Final[int] = 2

    def __init__(
        self,
        root: str,
        subset: str = "training",
        imgs_transform: Callable = None,
        msks_transform: Callable = None,
        download: bool = False,
        force: bool = False,
    ):
        """
        Args:
            root (str): Root directory for the BBBC039 dataset files.
            subset (str, optional): Dataset subset being loaded. Defaults to "training".
            imgs_transform (Callable, optional): Image transforms. Defaults to None.
            msks_transform (Callable, optional): Mask transforms. Defaults to None.
            download (bool, optional): Download flag for resources. Defaults to False.
            force (bool, optional): Force flag to overwrite dataset. Defaults to False.

        Raises:
            OSError: Root directory is not a directory or unmatched checksums.
            RuntimeError: Dataset argument is incorrect such as the data subset.
            FileNotFoundError: A required dataset directory is missing.
        """
        # Prepare, download and validate dataset
        super().__init__()
        self.root = Path(root)
        self._download(force) if download else None
        self._validate_directory()
        self.imgs_dir = self.root / self.resources.images.name
        self.msks_dir = self.root / self.resources.masks.name
        meta_dir = self.root / self.resources.metadata.name
        # Set image and mask filepaths
        if subset is not None and subset not in self.resources_subsets:
            # Capture invalid subset arguments
            raise RuntimeError(f"invalid subset '{subset}' -> {self.resources_subsets}")
        if subset is None:
            # Load in the entire dataset
            self.imgs = [self.imgs_dir / i for i in self.imgs_dir.iterdir()]
            self.msks = [self.msks_dir / i for i in self.msks_dir.iterdir()]
        else:
            # Load in the given dataset subset
            self.ids = [i.split(".")[0] for i in open(f"{meta_dir / subset}.txt", "r")]
            self.imgs = [self.imgs_dir / (i + self.images_extension) for i in self.ids]
            self.msks = [self.msks_dir / (i + self.masks_extension) for i in self.ids]
        # Preprocess the masks if not already completed
        self._preprocess_masks()
        # Set transforms
        self.imgs_transform = imgs_transform
        self.msks_transform = msks_transform

    def _preprocess_masks(self) -> None:
        # Define the new mask directory and check it exists or has correct checksum
        src_dirpath = self.msks_dir
        dst_dirpath = self.root / self.preprocessed_dirname
        if not dst_dirpath.exists():
            dst_dirpath.mkdir(exist_ok=True)
        if self.preprocessed_checksum == self._checksum_directory(dst_dirpath):
            self.msks = [dst_dirpath / i.name for i in self.msks]
            return
        # Process each masks file with a progress bar
        for mask_filepath in tqdm(
            src_dirpath.iterdir(),
            total=len([i for i in src_dirpath.iterdir()]),
            desc="Preprocessing masks",
            unit="img",
        ):
            # Define the masks filepath and skip if it exists
            src_path = src_dirpath / mask_filepath.name
            dst_path = dst_dirpath / mask_filepath.name
            if dst_path.exists():
                continue
            # Read mask and cut off channels and alpha, label nuclei and threshold ones
            mask = skimage.morphology.remove_small_objects(
                skimage.morphology.label(np.array(Image.open(src_path))[:, :, 0]),
                min_size=self.min_nuclei_size,
            )
            # Extract nuclei boundaries
            mask_boundaries = skimage.segmentation.find_boundaries(mask)
            for _ in range(2, self.nuclei_border_w, 2):
                mask_boundaries = skimage.morphology.binary_dilation(mask_boundaries)
            # Create the mask channels (background, nuclei, boundary)
            mask_label = np.zeros(mask.shape + (3,))
            mask_label[(mask == 0) & (mask_boundaries == 0), 0] = 1
            mask_label[(mask != 0) & (mask_boundaries == 0), 1] = 1
            mask_label[mask_boundaries == 1, 2] = 1
            # Save the new image
            Image.fromarray(mask_label.astype(np.uint8)).save(dst_path)
        # Update mask ids
        self.msks = [dst_dirpath / i.name for i in self.msks]

    def _validate_directory(self) -> bool:
        # Validate target dirpath exists and is a directory
        if not self.root.exists():
            raise FileNotFoundError(f"dataset directory missing: {self.root}")
        if not self.root.is_dir():
            raise OSError(f"dataset is not a valid directory: {self.root}")
        # Validate all required dataset directories in dirpath
        for resource in self.resources:
            dirname, (_, checksum) = resource.name, resource.value
            directory = self.root / dirname
            # Check directory exists and validate checksum
            if not directory.exists():
                raise FileNotFoundError(f"dataset directory missing: {dirname}")
            if not self._checksum_directory(directory) == checksum:
                raise OSError(f"dataset directory invalid checksum: {dirname}")
        return True

    def _checksum_directory(self, dirpath: Path) -> str:
        # Calculate dirpath hash checksum (hexdigest)
        hashdir = hashlib.md5()
        for filename in dirpath.iterdir():
            # Checksum filenames and file content
            hashdir.update(hashlib.md5(str(filename).encode()).digest())
            filepath = dirpath / filename
            hashdir.update(open(filepath, "rb")) if filepath.is_file() else None
        return hashdir.hexdigest()

    def _download(self, force: bool = False) -> None:
        # Download each dataset resource archive file
        self.root.mkdir(exist_ok=True)
        for resource in self.resources:
            # Check if the resource already exists
            resource_dir = self.root / resource.name
            if resource_dir.exists() and not force:
                continue
            # Request the resource to download
            resource_url = resource.value[0]
            resource_path = self.root / resource_url.split("/")[-1]
            resp = requests.get(resource_url, stream=True)
            # Process the resource request and write to the file
            with open(resource_path, "wb") as fp:
                resp_size = int(resp.headers["Content-Length"])
                for resp_chunk in tqdm(
                    resp.iter_content(chunk_size=self.download_chunksize),
                    desc=f"Downloading {str(resource_path)}",
                    unit="B",
                    total=math.ceil(resp_size / self.download_chunksize),
                    unit_scale=True,
                    unit_divisor=self.download_chunksize,
                ):
                    fp.write(resp_chunk)
            # Extract the downloaded resource and delete the resource
            with zipfile.ZipFile(resource_path, "r") as archive:
                archive.extractall(self.root)
            resource_path.unlink()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Load image and mask, convert to float for performing any further transforms
        img = np.array(Image.open(self.imgs[idx]), dtype=np.float_)
        msk = np.array(Image.open(self.msks[idx]), dtype=np.float_)
        # Apply transformations and coerce image and mask into tensors
        img = self.imgs_transform(img) if self.imgs_transform else img
        img = img if isinstance(img, torch.Tensor) else tv.transforms.ToTensor()(img)
        msk = self.msks_transform(msk) if self.msks_transform else msk
        msk = msk if isinstance(msk, torch.Tensor) else tv.transforms.ToTensor()(msk)
        return img, msk

    def __len__(self):
        # Dunder required for torch dataloaders
        return len(self.imgs)
