# Fruit Counting:
## Counting pineapples using Deep Learning with multispectral images.
This is a customized version of [EfficientDet](https://arxiv.org/abs/1911.09070) for our research about the importance multispectral images when dealing with object detection (EfficientDet paper: https://arxiv.org/abs/1911.09070). This package builds on top of [this](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) excellent EfficientDet PyTorch implementation. 
Also it encompases a strong pipeline to pre-process the multispectral images taken from a Unmanned Aerial Vehicle (UAV) model [DJI P4 Multispectral](https://www.dji.com/p4-multispectral). This pre-processing pipline follows the image processing guide: https://dl.djicdn.com/downloads/p4-multispectral/20200717/P4_Multispectral_Image_Processing_Guide_EN.pdf.

Requirements:

1. The total number of images in the dataset should not be larger than 10K, capacity should be under 5GB, and it should be free to download, i.e. baiduyun.
2. The dataset should be in the format of this repo.

## Install requirements (Python version 3.8).
    pip install -U scikit-learn
    pip install pycocotools numpy opencv-contrib-python tqdm tensorboard tensorboardX pyyaml webcolors
    pip install torch==1.4.0
    pip install torchvision==0.5.0
    pip install rasterio

## Data pre-processing
Normally, [DJI P4 Multispectral](https://www.dji.com/p4-multispectral) stores images in the following way:

        DJI_0010.JPG
        DJI_0011.TIF
        DJI_0012.TIF
        DJI_0013.TIF
        DJI_0014.TIF
        DJI_0015.TIF
        DJI_0016.JPG
        ...

The .JPG file represents de visible light image, and the other 5 .TIF files represent the multispectral bands of that specific capture. This pattern must be present in all datasets to pre-process since the script will look for the 6 images (visible-light, Red, Blue, Green, RedEdge, and NIR).   
The following line represents how the python file must be called to align all the images and apply all the photosensitivity between the multispectral images and sunlight sensor.

        # The param named start_numbering means the new unique number name that the image will have.  
        # This is important because DJI images name can be duplicated between directories from the same flight.
        python pre-processing/homography_align_ORB.py --dir_path /path/to/the/directory/where/your/images/are/located --results_dir_path /path/to/the/directory/where/the/aligned/images/will/be/located/ --start_numbering 1

After running the pre-processing step the multispectral images will look like this:

        00000001.JPG
        00000001.TIF
        00000002.JPG
        00000002.TIF
        ...

The ".TIF" file contains the 5 multispectral bands aligned in the following way: 1:Red, 2:Blue, 3:Green, 4:RedEdge, 5:NIR.  
The ".JPG" file corresponds to the original visible light image.
