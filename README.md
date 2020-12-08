# Homework-3 - Tiny Pascal VOC dataset

This model is for instance segmentation for the Tiny Pascal VOC dataset.
This repo is done with [this](https://github.com/facebookresearch/detectron2).

### Hardware
- Ubuntu 18.04.5 LTS
- Intel® Xeon® Silver 4210 CPU @ 2.20GHz
- NVIDIA GeForce RTX 2080 Ti

### Reproduce Submission
To reproduce my submission without training, do the following:
1. [Installation](#Installation)
2. [Data Preparation](#Data-Preparation)
3. [Inference](#Inference)

### Installation
This repo uses Detectron2 from facebook research team. Please install it first before running this repo.
I am using CUDA verion 10.2 and torch version 1.7.0. You can either run the following command or go to [Detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) and [Pytorch](https://pytorch.org/) to install.

`pip install torch torchvision`

`python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.7/index.html`

### Data Preparation
The data should be placed as follows:
```
repo
  +- train_images
  |  +- ...
  |
  +- test_images
  |  +- ...
  |
  +- output
  |  +- model_final.pth   (needed for inference)
  +- pascal_train.json
  +- test.json
  +- train.py
  +- infer.py
  +- X-101-32x8d.pkl   (needed for training)
  |  ...
```
The traing and testing images and thier repective annotation files can be downloaded [here](https://drive.google.com/drive/folders/1fGg03EdBAxjFumGHHNhMrz2sMLLH04FK).

### Training
To train, please download the pretrained weight [here](https://drive.google.com/file/d/1Q8tRJi7L8Dz2MKnnTZRrcQHixbAJTo_-/view?usp=sharing) and put it beside train.py. Simply run train.py. The weights should be saved in 'output' folder with name 'model_final.pth'. There will be several file saving weights at different training process. The batch_size is set to be 2. It can be changed in line 58, where 'cfg.SOLVER.IMS_PER_BATCH = batch_size'. Make it smaller if needed. The trained model will also be used to infer the test images after training and the prediction will also be made.

### Inference
for inference, please download the weights file [here](https://drive.google.com/file/d/1rFZY6UBCGqb9JECkEhgCOkKcvOKZnSTF/view?usp=sharing) and put it in output folder. Simply run infer.py and predictions.json will be created.

### Citation
[Detectron2](https://github.com/facebookresearch/detectron2)
