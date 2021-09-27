# Size Does Matter:
## The Importance of Scaling when Dealing with Limited Data in Object Detection

Our own version of [EfficientDet](https://github.com/google/automl/tree/master/efficientdet) for our research about the importance of size when dealing with homogeneous bounding box sizes (EfficientDet paper: <https://arxiv.org/abs/1911.09070>). This package builds on top of [this](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) excellent EfficientDet PyTorch implementation and the augmentation on the excellent [bbaug](https://github.com/harpalsahota/bbaug) library.

Requirements:

1. The total number of the image of the dataset should not be larger than 10K, capacity should be under 5GB, and it should be free to download, i.e. baiduyun.

2. The dataset should be in the format of this repo.


## Install requirements

    Python version 3.8
    pip install -U scikit-learn
    pip install pycocotools numpy opencv-python tqdm tensorboard tensorboardX pyyaml webcolors
    pip install torch==1.4.0
    pip install torchvision==0.5.0
    pip install bbaug

## Training

This sections shows the format of the dataset to be used in this framework.

### 1. Prepare your dataset

    # your dataset structure should be like this
    datasets/
        -your_project_name/
            -train_set_name/
                -*.jpg
            -val_set_name/
                -*.jpg
            -test_set_name/
                -*.jpg
            -annotations
                -instances_{train_set_name}.json
                -instances_{val_set_name}.json
                -instances_{test_set_name}.json
    
    # for example, our toy dataset 'apple'
    datasets/
        -apple/
            -train/
                -000000000001.jpg
                -000000000002.jpg
                -000000000003.jpg
            -val/
                -000000000004.jpg
                -000000000005.jpg
                -000000000006.jpg
            -test/
                -000000000004.jpg
                -000000000005.jpg
                -000000000006.jpg
            -annotations
                -instances_train.json
                -instances_val.json
                -instances_test.json

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

### 3.a. Train on coco from scratch(not necessary)

    # train efficientdet-d0 on coco from scratch 
    # with batchsize 12
    # This takes time and requires change 
    # of hyperparameters every few hours.
    # If you have months to kill, do it. 
    # It's not like someone going to achieve
    # better score than the one in the paper.
    # The first few epoches will be rather unstable,
    # it's quite normal when you train from scratch.
    
    python train.py -c 0 --batch_size 64 --optim sgd --lr 8e-2

### 3.b. Train a custom dataset from scratch

    # train efficientdet-d1 on a custom dataset 
    # with batchsize 8 and learning rate 1e-5
    
    python train.py -c 1 -p your_project_name --batch_size 8 --lr 1e-5

### 3.c. Train a custom dataset with pretrained weights (Highly Recommended)

    # train efficientdet-d2 on a custom dataset with pretrained weights
    # with batchsize 8 and learning rate 1e-3 for 10 epoches
    
    python train.py -c 2 -p your_project_name --batch_size 8 --lr 1e-3 --num_epochs 10 \
     --load_weights /path/to/your/weights/efficientdet-d2.pth
    
    # with a coco-pretrained, you can even freeze the backbone and train heads only
    # to speed up training and help convergence.
    
    python train.py -c 2 -p your_project_name --batch_size 8 --lr 1e-3 --num_epochs 10 \
     --load_weights /path/to/your/weights/efficientdet-d2.pth \
     --head_only True

### 4. Early stopping a training session

    # while training, press Ctrl+c, the program will catch KeyboardInterrupt
    # and stop training, save current checkpoint.

### 5. Resume training

    # let says you started a training session like this.
    
    python train.py -c 2 -p your_project_name --batch_size 8 --lr 1e-3 \
     --load_weights /path/to/your/weights/efficientdet-d2.pth \
     --head_only True
     
    # then you stopped it with a Ctrl+c, it exited with a checkpoint
    
    # now you want to resume training from the last checkpoint
    # simply set load_weights to 'last'
    
    python train.py -c 2 -p your_project_name --batch_size 8 --lr 1e-3 \
     --load_weights last \
     --head_only True

### 6. Evaluate model performance

    # eval on your_project, efficientdet-d5
    
    python coco_eval.py -p your_project_name -c 5 \
     -w /path/to/your/weights

## TODO

- [X] re-implement efficientdet
