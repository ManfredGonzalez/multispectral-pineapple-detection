from bbaug.policies import policies
import argparse
import os
import sys
import glob
from shutil import copyfile
import cv2
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import boolean_string

def get_args():
    """Get all expected parameters"""
    parser = argparse.ArgumentParser('Resize YOLOv3 datasets to zoom-in or zoom-out')
    parser.add_argument('--input_path', type=str, default='datasets/yolo_format/apple_yolov4pytorch')
    parser.add_argument('--classes_file', type=str, default='classes.txt')
    parser.add_argument('--imgs_extension', type=str, default='JPG')
    parser.add_argument('--output_path', type=str, default='datasets/yolo_format/apple_yolov4pytorch_resized')

    args = parser.parse_args()
    return args

def yoloFileToList(annotations_path,imageHeight, imageWidth):
    labels = []
    boxes = []
    yoloFile = open(annotations_path,'r').readlines()
    for annotation in yoloFile:
        class_id,x1,y1,x2,y2 = annotation.split()

        xcenter = float(x1) * imageWidth
        ycenter = float(y1) * imageHeight
        bbox_width = float(x2) * imageWidth
        bbox_height = float(y2) * imageHeight
        x_top_left = int(abs(xcenter-(bbox_width/2)))
        y_top_left  = int(abs(ycenter - (bbox_height/2)))
        x_bottom_right = x_top_left + bbox_width
        y_bottom_right = y_top_left + bbox_height

        class_id,x1,y1,x2,y2 = int(class_id),x_top_left, y_top_left, x_bottom_right, y_bottom_right
        labels.append(class_id)
        boxes.append([x1,y1,x2,y2])
    return labels,boxes

def fromListsToSaveYoloFile(boxes,filePath,imageHeight, imageWidth):
    
    for label,x1,y1,x2,y2 in boxes:
        bbox_width = abs(x1-x2)
        bbox_height = abs(y1-y2)
        xcenter = x1 + (bbox_width/2)
        ycenter = y1 + (bbox_height/2)

        with open(filePath, "a") as my_file: 
            my_file.write(f'{label} {round((xcenter/imageWidth),6)} {round((ycenter/imageHeight),6)} {round((bbox_width/imageWidth),6)} {round((bbox_height/imageHeight),6)}\n')


def resize_dataset(opt):
    myImages = glob.glob(f'{opt.input_path}/*.{opt.imgs_extension}')
    myAnnots = glob.glob(f'{opt.input_path}/*.txt')

    bands = ['Red','Green','Blue','RedEdge','NIR'] 
    zoom_in_policy = policies.policies_pineapple(1.1)
    zoom_in_policy_container = policies.PolicyContainer(zoom_in_policy, random_state = None)

    for image_path,annotations_path in tqdm(zip(myImages,myAnnots)):
        image = cv2.imread(image_path)
        labels,boxes = yoloFileToList(annotations_path,image.shape[0],image.shape[1])
        random_policy = zoom_in_policy_container.select_random_policy()
        img_aug, bbs_aug = zoom_in_policy_container.apply_augmentation(
                random_policy,
                image,
                boxes,
                labels,
            )
        fromListsToSaveYoloFile(bbs_aug,os.path.join(opt.output_path,os.path.basename(annotations_path)),image.shape[0],image.shape[1])
        cv2.imwrite(os.path.join(opt.output_path,os.path.basename(image_path)),img_aug)
        for band in bands:
            base_name = os.path.splitext(image_path)[0]
            band_image = cv2.imread(f'{base_name}_{band}.TIF',cv2.IMREAD_GRAYSCALE)
            img_aug, _ = zoom_in_policy_container.apply_augmentation(
                random_policy,
                band_image,
                boxes,
                labels,
            )
            cv2.imwrite(os.path.join(opt.output_path,f'{os.path.basename(base_name)}_{band}.TIF'),img_aug)


if __name__ == '__main__':
    opt = get_args()
    if not os.path.exists(opt.output_path):
                os.makedirs(opt.output_path)
    resize_dataset(opt)
    print('Done')