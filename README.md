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

`python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.7/index.html`

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
To train, please download the pretrained weight [here](https://drive.google.com/file/d/16JwAXozDpqjoWRE1MySUKH0JdkstXMqf/view?usp=sharing) and put it beside train.py. Simply run train.py. The weights should be saved in 'logs/numbers' folder. There will be several file saving weights at different training process. The batch_size is set to be 3. Make it smaller in numbers.yml if memory is not sufficent.

### Inference
for inference, please download the weights file [here](https://drive.google.com/file/d/1t9W1wxUjfBOTtGo0I7-tlpgdlKqvSM01/view?usp=sharing) and put it beside infer.py. Simply run infer.py and predictions.json containing images file names and their corresponding predictions will be created.

### Citation
[Yet Anothor EfficientDet Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)
