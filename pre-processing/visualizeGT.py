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
        boxes.append([class_id,x1,y1,x2,y2])
    return boxes

def printConflictsSolutionROIS(rois,imageToPrint,obj_list):
  for label in rois:
    #print(labelsarray)
      ## Translating the slice coordinates to the original image
    cv2.rectangle(imageToPrint, (int(label[1]), int(label[2])), (int(label[3]), int(label[4])), (255, 0, 0), 2)
    cv2.putText(imageToPrint, '{}'.format(obj_list[label[0]]),
                                (int(label[1]), int(label[2])), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 0), 1)
  #plt.figure(figsize=(16, 10))
  #plt.imshow(imageToPrint)
  #plt.show()
  return imageToPrint


def resize_dataset(opt):
    myImages = glob.glob(f'{opt.input_path}/*.{opt.imgs_extension}')
    myAnnots = glob.glob(f'{opt.input_path}/*.txt')

    myAnnots = [annot for annot in myAnnots if os.path.basename(annot)!='classes.txt']

    bands = ['Red','Green','Blue','RedEdge','NIR'] 
    obj_list = ['pineapple']

    for image_path,annotations_path in tqdm(zip(myImages,myAnnots)):
        image = cv2.imread(image_path)
        boxes = yoloFileToList(annotations_path,image.shape[0],image.shape[1])
        image = printConflictsSolutionROIS(boxes,image,obj_list)
        cv2.imwrite(os.path.join(opt.output_path,os.path.basename(image_path)),image)
        for band in bands:
            base_name = os.path.splitext(image_path)[0]
            band_image = cv2.imread(f'{base_name}_{band}.TIF',cv2.IMREAD_GRAYSCALE)
            band_image = printConflictsSolutionROIS(boxes,band_image,obj_list)
            cv2.imwrite(os.path.join(opt.output_path,f'{os.path.basename(base_name)}_{band}.TIF'),band_image)


if __name__ == '__main__':
    opt = get_args()
    if not os.path.exists(opt.output_path):
                os.makedirs(opt.output_path)
    resize_dataset(opt)
    print('Done')