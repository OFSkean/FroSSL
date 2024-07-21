# FroSSL: Frobenius Norm Minimization for Efficient Multiview Self-Supervised Learning

This is the official PyTorch implementation of the [FroSSL paper](https://arxiv.org/pdf/2310.02903):

```
@article{skean2023frossl,
  title={FroSSL: Frobenius Norm Minimization for Self-Supervised Learning},
  author={Oscar Skean and Aayush Dhakal and Nathan Jacobs and Luis Gonzalo Sanchez Giraldo},
  journal={arXiv preprint arXiv:2310.02903},
  year={2023}
}
```
This implementation started as a fork of the fantastic [solo-learn](https://github.com/vturrisi/solo-learn.git) library. We are currently working on a pull request to merge our contributions into the library.

## Preparation
### Installing Requirements
We provide instructions for how to setup a conda environment and install the necessary dependencies:

```
# clone the repository
git clone git@github.com:OFSkean/frossl.git
cd ./frossl

# create env
conda create -n frossl python=3.10
conda activate frossl

# install dependencies
pip install -r requirements.txt
```

### Datasets
The datasets CIFAR-10, CIFAR-100, and STL-10 are able to be downloaded automatically by Pytorch. If you want to run on other datasets like tiny-imagenet, some preparation will be required. By default, the data is assumed to live in `./datasets/{dataset-name}`. To change this, you have to adjust the configuration files which can be found in `./scripts/pretrain/*`.

#### Tiny ImageNet
We have provided an installation script for Tiny ImageNet that can be used like:

```
cd scripts/utils/tiny-imagenet
./downloader.sh
```

#### ImageNet
If you don't already have ImageNet downloaded, we recommend following [this guide](https://cloud.google.com/tpu/docs/imagenet-setup#download-dataset). Note that ImageNet **must be downloaded** to prepare the ImageNet-100 dataset.

#### ImageNet-100
Once you have the ImageNet dataset downloaded, you can create the ImageNet-100 dataset with:

```
python make_imagenet100.py full/imagenet/path desired/imagenet100/path
```

## Training and Evaluating a Model
1. Make sure you have a wandb account and are logged in via the CLI. All hyperparameters, losses, system details, etc. will get logged to wandb.

2. Check out the .yaml configuration at `./scripts/pretrain/stl10/frossl.yaml`. This file is where you can tweak hyperparameters for the training procedure, augmentations, and loss. Every dataset and method has its own configuration file. By default, these configurations are setup to match what we used for the paper.

3. We have provided three scripts to serve as examples for training a model: **run_cifar10.sh**, **run_imagenet100.sh**, **run_tiny.sh**. These are currently configured to run FroSSL on a specific dataset. 

4. Run an above script (or modify it) like  `bash ./run_cifar10.sh`


## Objective Function
The FroSSL objective function is [implemented here](https://github.com/OFSkean/FroSSL/blob/main/solo/losses/frossl.py) and [used here](https://github.com/OFSkean/FroSSL/blob/main/solo/methods/frossl.py).

## Let Us Know
Please open an issue on Github if you encounter any errors or difficulties using this implementation. We are happy to help resolve issues or answer any questions you may have!
