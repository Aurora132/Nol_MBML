# Material composition characterization from computed tomography via self-supervised learning promotes disease diagnosis


## Contents

- [Overview](#overview)
- [Repo Contents](#repo-contents)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Instructions for Use](#instructions-for-use)
- [License](./LICENSE)
- [Citation](#citation)

# 1. Overview

This project provides a self-supervised learning approach for performing multi-material decomposition from dual-energy CT data without explicitly annotated labels. Using the code requires users to have basic knowledge about python, PyTorch, and deep neural networks.

# 2. Repo Contents
- [decomposition_initialization.py](./decomposition_initialization.py): pseudo label generation.
- [networks/unet_ConvNeXt.py](./networks/unet_ConvNeXt.py): network definition of proposed method.
- [utils.py](./utils.py): util functions and loss functions definition.
- [data_loader.py](./data_loader.py): data loader definition for model training.
- [train_pre.py](./train_pre.py): main code for the first training stage.
- [train_sec.py](./train_sec.py): main code for the second training stage.
- [train_sec.py](./train_sec.py): main code for the second training stage.
- [train_SECT_DECT.py](./train_SECT_DECT.py): main code for training the DECT imaging generation module.
- [decompose_demo.py](./decompose_demo.py): decompose the demo DECT data into four basic materials (Adipose, Air, Muscle, and Iodine).
- You can perform other material decomposition by changing the weights and the input data here.
- [checkpoints/SECT_DECT.pth](./checkpoints/SECT_DECT.pth): trained weights for DECT data generation.
- [checkpoints/decompose_four.pth](./checkpoints/decompose_four.pth): trained weights for decomposing DECT data into four basic materials (Adipose, Air, Muscle, and Iodine).
