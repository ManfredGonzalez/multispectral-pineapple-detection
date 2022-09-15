# Fruit Counting:
## Counting pineapples using Deep Learning with multispectral images.
This is a customized version of [EfficientDet](https://arxiv.org/abs/1911.09070) for our research about the importance multispectral images when dealing with object detection (EfficientDet paper: https://arxiv.org/abs/1911.09070). This package builds on top of [this](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) excellent EfficientDet PyTorch implementation. In the other hand, [YOLOv5](https://github.com/ultralytics/yolov5) is another Object Detection approach we used to train with our datasets.
Also it encompases a strong pipeline to pre-process the multispectral images taken from a Unmanned Aerial Vehicle (UAV) model [DJI P4 Multispectral](https://www.dji.com/p4-multispectral). This pre-processing pipline follows the image processing guide: https://dl.djicdn.com/downloads/p4-multispectral/20200717/P4_Multispectral_Image_Processing_Guide_EN.pdf.

Requirements:

1. The total number of images in the dataset should not be larger than 10K, capacity should be under 5GB, and it should be free to download, i.e. baiduyun.
2. The dataset should be in the format of this repo.

## Install requirements (Python version 3.8).
    pip install -U scikit-learn
    pip install pycocotools numpy opencv-contrib-python tqdm tensorboard tensorboardX pyyaml webcolors
    pip install torch==1.4.0
    pip install torchvision==0.5.0

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
        python pre-processing/homography_align_SIFT.py --dir_path /path/to/the/directory/where/your/images/are/located --results_dir_path /path/to/the/directory/where/the/aligned/images/will/be/located/ --start_numbering 1

After running the pre-processing step the multispectral images will look like this:

        00000001.JPG
        00000001_Blue.TIF
        00000001_Green.TIF
        00000001_NIR.TIF
        00000001_Red.TIF
        00000001_RedEdge.TIF
        00000002.JPG
        00000002_Blue.TIF
        00000002_Green.TIF
        00000002_NIR.TIF
        00000002_Red.TIF
        00000002_RedEdge.TIF
        ...
 
The ".JPG" file corresponds to the original visible light image.
The ".TIF" files are the 5 multispectral bands aligned taking the visible light as reference. 

## Training

This sections shows the format of the dataset to be used in this framework.

### 1. Prepare your dataset

    # your dataset structure should be like this
    datasets/
        -your_project_name/
            -train_set_name/
                -*.JPG
                -*_Blue.TIF
                -*_Green.TIF
                -*_NIR.TIF
                -*_Red.TIF
                -*_RedEdge.TIF
            -val_set_name/
                -*.JPG
                -*_Blue.TIF
                -*_Green.TIF
                -*_NIR.TIF
                -*_Red.TIF
                -*_RedEdge.TIF
            -test_set_name/
                -*.JPG
                -*_Blue.TIF
                -*_Green.TIF
                -*_NIR.TIF
                -*_Red.TIF
                -*_RedEdge.TIF
            -annotations
                -instances_{train_set_name}.json
                -instances_{val_set_name}.json
                -instances_{test_set_name}.json
                
### 2. Manual set project's specific parameters

    # create a yml file {your_project_name}.yml under 'projects'folder 
    # modify it following 'coco.yml'

    # for example
    project_name: coco
    train_set: train2017
    val_set: val2017
    num_gpus: 4  # 0 means using cpu, 1-N means using gpus 

    # mean and std in RGB order, actually this part should remain unchanged as long as your dataset is similar to coco.
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]

    # this is coco anchors, change it if necessary
    anchors_scales: '[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]'
    anchors_ratios: '[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]'

    # objects from all labels from your dataset with the order from your annotations.
    # its index must match your dataset's category_id.
    # category_id is one_indexed,
    # for example, index of 'car' here is 2, while category_id of is 3
    obj_list: ['person', 'bicycle', 'car', ...]
    
### 3.a. Train a custom dataset from scratch

    # train efficientdet-d1 on a custom dataset 
    # with batchsize 8 and learning rate 1e-5
    
    python train.py -c 1 -p your_project_name --batch_size 8 --lr 1e-5 --bands_to_apply 'Red Green Blue RedEdge NIR'

### 3.b. Train a custom dataset with pretrained weights (Highly Recommended)

    # train efficientdet-d2 on a custom dataset with pretrained weights
    # with batchsize 8 and learning rate 1e-3 for 10 epoches
    
    python train.py -c 2 -p your_project_name --batch_size 8 --lr 1e-3 --num_epochs 10 \
     --load_weights /path/to/your/weights/efficientdet-d2.pth --bands_to_apply 'Red Green Blue RedEdge NIR'
    
    # with a coco-pretrained, you can even freeze the backbone and train heads only
    # to speed up training and help convergence.
    
    python train.py -c 2 -p your_project_name --batch_size 8 --lr 1e-3 --num_epochs 10 \
     --load_weights /path/to/your/weights/efficientdet-d2.pth \
     --bands_to_apply 'Red Green Blue RedEdge NIR' \
     --head_only True

### 4. Early stopping a training session

    # while training, press Ctrl+c, the program will catch KeyboardInterrupt
    # and stop training, save current checkpoint.

### 5. Resume training

    # let says you started a training session like this.
    
    python train.py -c 2 -p your_project_name --batch_size 8 --lr 1e-3 \
     --load_weights /path/to/your/weights/efficientdet-d2.pth \
     --bands_to_apply 'Red Green Blue RedEdge NIR' \
     --head_only True
     
    # then you stopped it with a Ctrl+c, it exited with a checkpoint
    
    # now you want to resume training from the last checkpoint
    # simply set load_weights to 'last'
    
    python train.py -c 2 -p your_project_name --batch_size 8 --lr 1e-3 \
     --load_weights last \
     --bands_to_apply 'Red Green Blue RedEdge NIR' \
     --head_only True

### 6. Evaluate model performance

    # eval on your_project, efficientdet-d5
    
    python coco_eval.py -p your_project_name -c 5 \
     --bands_to_apply 'Red Green Blue RedEdge NIR' \
     -w /path/to/your/weights
