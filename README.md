# **SIA-Net: Spatial- and Intensity-Aware Network with AlsEMA for Semi-Supervised 3D Medical Image Segmentation**

## Overview

![1775235350800](image/2025.0707/1775235350800.jpg)

- A spatial- and intensity-aware network for semi-supervised image segmentation.
- Image separation mainly focuses on spatial- and intensity-awareness.
- Adaptive loss-sensitive exponential moving average update mechanism.
- Experiments and evaluation on four different datasets.

## Usage

### Requirements

The recommend python and package version:

- python>=3.10.0
- pytorch>=1.13.1

### Train

here we use an example(Traning 3D Unet) to teach you how use this repository

```
python train.py config=SIABETA
```

### Predict

run the code

```
python predict.py config=SIABETA config.ckpt=XXX
```
