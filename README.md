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
- [train_SECT_DECT.py](./train_SECT_DECT.py): main code for training the DECT imaging generation module.
- [decompose_demo.py](./decompose_demo.py): decompose the demo DECT data into four basic materials (Adipose, Air, Muscle, and Iodine).
- You can perform other material decomposition by changing the weights and the input data here.
- [checkpoints/SECT_DECT.pth](./checkpoints/SECT_DECT.pth): trained weights for DECT data generation. [Download link](https://1drv.ms/u/s!Al8vukYxw_dFgzyl33BEYzpgddcH?e=iHeWYA)
- [checkpoints/decompose_four.pth](./checkpoints/decompose_four.pth): trained weights for decomposing DECT data into four basic materials (Adipose, Air, Muscle, and Iodine). [Download link](https://1drv.ms/u/s!Al8vukYxw_dFgz3BxKMYQHynvANQ?e=ilKqyp)


# 3. System Requirements

## Prerequisites
- Ubuntu 18.04
- NVIDIA GPU + CUDA (Geforce RTX 3090 with 24GB memory, CUDA 11.4)

## Package Versions
- python 3.8
- pytorch 1.10.1
- torchvision 0.11.2
- opencv-python 4.6.0.66
- numpy 1.19.5

# 4. Instructions for Use

## Training
- Run `python train_pre.py` to perform the first training process with the default setting. Then change the parameter `pretrain_dir` and run `python train_sec.py` to begin the second training process.
- Run `python train_SECT_DECT.py` to start the DECT imaging generation module training stage.

## Test demo
Run `python decompose_demo.py` to test the trained model deposited in `./checkpoints/decompose_four/` on the data in `./data/demo_data/`. Image results are stored in `./decompose_result/`.

# 5. License
This project is covered under the BSD-3-Clause License.
