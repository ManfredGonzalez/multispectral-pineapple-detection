# Fruit Counting:
## Counting pineapples using Deep Learning with multispectral images.
This is a customized version of [EfficientDet](https://arxiv.org/abs/1911.09070) for our research about the importance multispectral images when dealing with object detection (EfficientDet paper: https://arxiv.org/abs/1911.09070). This package builds on top of [this](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) excellent EfficientDet PyTorch implementation. 
Also it encompases a strong pipeline to pre-process the multispectral images taken from a Unmanned Aerial Vehicle (UAV) model [DJI P4 Multispectral](https://www.dji.com/p4-multispectral). This pre-processing pipline follows the image processing guide: https://dl.djicdn.com/downloads/p4-multispectral/20200717/P4_Multispectral_Image_Processing_Guide_EN.pdf.

Requirements:

1. The total number of the image of the dataset should not be larger than 10K, capacity should be under 5GB, and it should be free to download, i.e. baiduyun.
2. The dataset should be in the format of this repo.


## Install requirements (Python version 3.8).
    pip install -U scikit-learn
    pip install pycocotools numpy opencv-python tqdm tensorboard tensorboardX pyyaml webcolors
    pip install torch==1.4.0
    pip install torchvision==0.5.0
    pip install rasterio

## Data pre-processing
Normally, [DJI P4 Multispectral](https://www.dji.com/p4-multispectral) stores the pictures in the following way:
    DJI_0010.JPG
    DJI_0011.TIF
    DJI_0012.TIF
    DJI_0013.TIF
    DJI_0014.TIF
    DJI_0015.TIF
    DJI_0016.JPG
    ...
