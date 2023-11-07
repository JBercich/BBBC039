# BBBC039: Nuclei Instance Segmentation

## Experiment Summary

Experimental analysis of UNet-DLA instance segmentation models and segmentation-refined RandAugment. Ablations seek to assert effective segmentation performance supported by a simple augmentation strategy with reduced need for architecture ablations for reduced experimental iteration development, improved automation capacity, and greater multi-task capacity. Experimental insights target limitations in low-scale dataset availability and diversity through minimally complex uniquely refined augmentation policies. Lowly-modified encoder-decoder architecture performance from U-Net with Deep Layer Aggregation can support better performance results without testing against multiple architectures and obfuscated rapid iterative experiementation across more research settings in the biomedical field. The primary key insights from this experiment include:

- Augmentation policies have greater potential for more diverse multiclass challenges and yield higher test performance.

- Simple class features are more rapidly learned with the applied RandAugment strategy for all ablations.

- Unique hyperparameter trade-off between introduced augmentation magnitude and network size must be considered in experimental settings; different from general overparameterisation issues where regularisation penalties are poorly captured.

- Second-best literature (at this time) was outperformed under binary semantic segmentation for final test set Dice coefficient (98.77; +2.337) and AJI (97.96; +7.76), where 3-class instance segmentation achieved near equal success.

## Further Results

Beyond the reproducible implementation provided in the `bbbc039` directory using PyTorch, a [concise report](report/main.pdf) was written (6 pages), and all results were logged on a public [WnB experiment project](https://wandb.ai/joshbercich/BBBC039) where appropriate plots and artifacts can be accessed.

- Inference examples during the training process of two RandAugment models for the medium network size. Images are from different training epochs.

![Epoch inference images](figures/epoch-inference.png | width=70)

- Attached ground truth labels to the previous model at each respectively logged epoch.

![Epoch label images](figures/epoch-labels.png | width=70)

- Example RandAugment augmentations performed for varying hyperparameters *n*, the number of applied operations, and *m*, the magnitude of each operation.

![RandAugment examples](figures/randaugment.png | width=80)

## Usage

### Dependecy Management

Project dependencies are managed using `poetry` as defined by the `pyproject.toml` file. Python versions are restricted to `>=3.10.12,<3.13`. For install the required dependencies from the project root directory:

```bash
# Dependency installation
$ pip install poetry
$ poetry install
$ poetry run python --version
```

### Hyperparameter Experiment

The reproducible parameter script is defined in `bbbc039/train.py`. The `lightning` package is used to create model and dataset modules which are specifically defined for this experimental pipeline. `RandAugment` and `BBBC039` classes from the `bbbc039/dataset` directory describe more specific operations on the BBBC039 segmentation dataset to align with more common instance segmentation datasets. Future work may seek to refine the data format to align with the `ultralytics` library for smaller storage requirements and runtime masking. For running the experiment:

```bash
# Ignore `datapath` for automatic dataset download
$ python train.py --datapath=<path_to_dataset>
```
