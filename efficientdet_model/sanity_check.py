import json
import os
import numpy as np
import psutil
import csv
import shutil
import sys

import argparse
import torch
import yaml
import cv2
import rasterio

from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess_ml, invert_affine, postprocess_original, boolean_string
sys.path.append("efficientdet_model/metrics/")
from metrics.mean_avg_precision import mean_average_precision

def get_rois_from_gtjson(coco_json,img_id,image):
    '''
    Extract the ground truth bounding boxes and store the results into a list. Important: The ground truth score must be always 1 by default 
    indicating a 100% of confidence in the object existence.

    Params
    :coco_json (pycocotools.coco) -> pycocotools for loading the bboxes from the json.

    Return
    :(list) -> [['image_id': int,'category_id': int,'score': 1, xmin: int, ymin: int, xmax: int, ymax: int], [...]]    
    '''
    ground_truth_boxes = []
    image_ids = coco_json.getImgIds()
    ann_ids = coco_json.getAnnIds(imgIds=img_id)
    coco_annotations = coco_json.loadAnns(ann_ids)
    for i in range(len(coco_annotations)):
        label = coco_annotations[i]['category_id']
        xmin = coco_annotations[i]['bbox'][0]
        ymin = coco_annotations[i]['bbox'][1]
        xmax = xmin + coco_annotations[i]['bbox'][2]
        ymax = ymin + coco_annotations[i]['bbox'][3]
        ground_truth_boxes.append([img_id,label,1, xmin, ymin, xmax, ymax])
        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)            
    
    return ground_truth_boxes

    



def run_metrics(compound_coef, 
                nms_threshold, 
                confidence_threshold, 
                use_cuda,  
                project_name, 
                weights_path, 
                max_detect_list,
                input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536],
                #input_sizes = [1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280],
                metric_option='coco',
                set_to_use='test_set',
                num_of_workers=0,
                batch_size=2,
                use_only_vl=False,
                bands_to_apply = None,
                use_normalization = True):    

    '''
    Method to perform the calculation of the metrics.
    
    :set_to_use (str) -> name of the set that is going to be used. E.g. test, val, or train.
    :conf_threshold (float) -> confidence threshold to filter results.
    :nms_threshold (float) -> non-maximum supression to filter results.
    :input_sizes (list<int>) -> input sizes of the different architectures of EfficientDet.
    :compound_coef (int) -> compound coefficient that indicates the architecture used.
    :use_cuda (bool) -> use gpu or not.
    :project_name (str) -> name of the .yml file the same as de dataset
    :orig_height (int) -> current height of the image
    :dest_height (int) -> resulted height of the transformed images
    :metric_option (str) -> indicate which metric will be used (coco, simple). The coco metric uses
                            101 point interpolation to calculate the average precision while the simple 
                            option uses 11 point interpolation.
    :num_of_workers (int) -> number of workers for the dataloader of the dataset
    :batch_size (int) -> size of the batch for the dataloader of the dataset
    :augment_dataset (bool) -> apply scaling transformation or not
    :weights_path (str) -> path of the trained weights
    '''


    #load default values, parameters and initialize
    #---------------------------------------------------------------------------------------------------------
    params = yaml.safe_load(open(f'projects/{project_name}.yml'))
    obj_list = params['obj_list']
    
    SET_NAME = params[set_to_use]
    dataset_json = f'datasets/{params["project_name"]}/annotations/instances_{SET_NAME}.json'
    dataset_imgs_path = f'datasets/{params["project_name"]}/{SET_NAME}/'

    # load data set
    coco = COCO(dataset_json)
    image_ids = coco.getImgIds()
    
    # get the number of bboxes from the ground truth
    groundtruth_num = 0
    with open(dataset_json, "r") as read_file:
        data = json.load(read_file)
        groundtruth_num = len(data["annotations"])

    
   

    if not use_only_vl:
        bands_to_apply = [item for item in bands_to_apply.split(' ')]
        in_channels = len(bands_to_apply)
    else:
        bands_to_apply = None
        in_channels = 3  
    

          
    #run the prediction of the bounding boxes and store results into a file
    # iterate over every image
    for image_id in tqdm(image_ids):
        image_info = coco.loadImgs(image_id)[0]
        image_path = dataset_imgs_path + image_info['file_name']
        jpg_image_path = image_path
        
        visible_light = cv2.imread(image_path)


        image_name= os.path.splitext(image_info['file_name'])[0]
        filename, file_extension = os.path.splitext(image_path)
        image_path = f'{image_path[:-len(file_extension)]}.TIF'
        dictionary = ['Red','Green','Blue','RedEdge','NIR']  
        for bandName in dictionary:
            image = cv2.imread(f'{filename}_{bandName}.TIF')
            ground_truth_boxes = get_rois_from_gtjson(coco,image_id,image)
            cv2.imwrite(f'datasets/{params["project_name"]}/sanity_check/{image_name}_{bandName}.TIF',image)
        ground_truth_boxes = get_rois_from_gtjson(coco,image_id,visible_light)
        cv2.imwrite(f'datasets/{params["project_name"]}/sanity_check/{image_name}_visible_light.jpg',visible_light)
    

#               Section for handling parameters from user
#--------------------------------------------------------------------------------------------------------------------
def throttle_cpu(cpu_list):
    """Read file with parameters"""
    p = psutil.Process()
    for i in p.threads():
        temp = psutil.Process(i.id)
        temp.cpu_affinity([i for i in cpu_list])


def get_args():
    """Get all expected parameters"""
    parser = argparse.ArgumentParser('EfficientDet Pytorch - Evaluate the model')
    parser.add_argument('-p', '--project', type=str, default="")
    parser.add_argument('-c', '--compound_coef', type=int, default=0)
    parser.add_argument('-n', '--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('-w', '--weights', type=str, default="") 
    parser.add_argument('--nms_thres', type=float, default=0.5)
    parser.add_argument('--conf_thres', type=float, default=0.5)
    parser.add_argument('--use_cuda', type=boolean_string, default=False)
    parser.add_argument('--max_detect', type=str, default="10000")  
    parser.add_argument('--debug', type=boolean_string, default=False)
    parser.add_argument('--metric', type=str, default="simple")
    parser.add_argument('--bands_to_apply', type=str, default="1_2_3")
    parser.add_argument('--use_only_vl', type=boolean_string, default=False)
    parser.add_argument('--input_sizes', type=str, default="512 640 768 896 1024 1280 1280 1536 1536")

    args = parser.parse_args()
    return args


#main method to be called
if __name__ == '__main__':
    #throttle_cpu([28,29,30,31,32,33,34,35,36,37,38,39]) 
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    opt = get_args()

    # get the values from the string
    max_detections = [int(item) for item in opt.max_detect.split(' ')]
    input_sizes = [int(item) for item in opt.input_sizes.split(' ')]
    # main method to measure performance
    run_metrics(opt.compound_coef, 
                opt.nms_thres, 
                opt.conf_thres,
                opt.use_cuda,  
                opt.project, 
                opt.weights,   
                max_detections,
                metric_option=opt.metric,
                use_only_vl=opt.use_only_vl,
                bands_to_apply=opt.bands_to_apply,
                input_sizes = input_sizes)
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    #test_case1()
