# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import os
import sys
import csv
from pathlib import Path

import random

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

sys.path.append("yolov5/metrics/")
from metrics.mean_avg_precision import mean_average_precision

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync
from utils.utils import boolean_string

def eval_fh(pineapples_detected, 
            ground_truth_boxes, 
            iou_threshold, 
            num_classes):
    '''
    Method to calculate precision, recall, and average precision with a general approach, i.e., not using 
    number of max detections (as performed by pycocotools). This method is a single general calculation.

    Params
    :pineapples_detected (list) -> Format:[['image_id': int,'category_id': int,'score': 1, xmin: int, ymin: int, xmax: int, ymax: int], [...]]
    :ground_truth_boxes (list) -> Format: [['image_id': int,'category_id': int,'score': 1, xmin: int, ymin: int, xmax: int, ymax: int], [...]] 
    :iou_threshold (int) -> iou to be used to filter out results.
    :num_classes (int) -> number of total classes (categories) we have.
    '''
    p,r,ap = mean_average_precision(pineapples_detected, 
                                    ground_truth_boxes, 
                                    iou_threshold, 
                                    box_format="corners", 
                                    num_classes=num_classes)
    return p,r,ap

@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # "path/to/Red/model/weights.pt path/to/Green/model/weights.pt path/to/Blue/model/weights.pt path/to/RedEdge/model/weights.pt path/to/NIR/model/weights.pt"
        dataset_path=ROOT / 'data/',  # file/dir/to/dataset
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.4,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        bands_to_apply = None,  #Red Green Blue RedEdge NIR 
        csv_file_name = 'results'
        ):
    
    dataset_path = str(dataset_path)
    source = os.path.join(dataset_path,'images')
    source_gt = os.path.join(dataset_path,'labels')

    if len(bands_to_apply.strip())!=0:
        bands_to_apply = bands_to_apply.split(' ')
        new_bands = []
        for band in bands_to_apply:
            if '_' in band:
                band = band.split('_')
                new_bands.append(band)
            elif band.lower() == 'rgb':
                new_bands.append(None)
            else:
                new_bands.append([band])
        bands_to_apply = new_bands

        #model.yaml['ch'] = len(bands_to_apply)
    else:
        bands_to_apply = None
        #model.yaml['ch'] = 3
    # Directories
    #save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir = Path('yolov5/results')
    #(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    
    cvs_path = f'{save_dir.resolve()}/{csv_file_name}.csv'
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    '''if not Path(cvs_path).exists():
        with open(cvs_path, "a") as myfile:
            my_writer = csv.writer(myfile, delimiter=',', quotechar='"')
            my_writer.writerow(["bands_used", "groundtruth_num", "num_detections", "nms_threshold", "confidence_threshold", "precision", "recall", "f1_score","weights_path"])'''

    # Initialize
    #set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load models
    #models_weights
    if len(weights[0].strip())!=0:
        models_weights = [item for item in weights[0].split(' ')]
        #model.yaml['ch'] = len(bands_to_apply)
    else:
        raise Exception('The weights of the models are required')
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    inferences = {}
    detections_list = []
    ground_truth_dict = {}
    ground_truth_list = []
    for j in range(len(models_weights)):
        print(f'Processing model: {models_weights[j]}')
        print(f'Using bands: {bands_to_apply[j]}')
        model = attempt_load(models_weights[j], map_location=device)
    
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    
    
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        pt = True
        # Dataloader
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, bands_to_apply = bands_to_apply[j])
        bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        if pt and device.type != 'cpu':
            if bands_to_apply:
                model(torch.zeros(1, len(bands_to_apply), *imgsz).to(device).type_as(next(model.parameters())))  # run once
            else:
                model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once
        dt, seen = [0.0, 0.0, 0.0], 0

        
        for (path, img, im0s, vid_cap) in tqdm(dataset):
            ## collect the groundtruth
            image_name = Path(path).stem
            if len(ground_truth_dict) < len(dataset.files):
                gt_path = Path(os.path.join(source_gt,f'{image_name}.txt'))
                if not gt_path.is_file():
                    raise Exception(f'Image {image_name} has not have the required ground truth file')
                gt_list = open(gt_path, "r").read().split('\n')
                img_gt = []
                for i in range(len(gt_list)):
                    label = gt_list[i].split(' ')
                    if len(label)==5:
                        xcenter = float(label[1]) * im0s.shape[1]
                        ycenter = float(label[2]) * im0s.shape[0]
                        bbox_width = float(label[3]) * im0s.shape[1]
                        bbox_height = float(label[4]) * im0s.shape[0]
                        x_top_left = int(abs(xcenter-(bbox_width/2)))
                        y_top_left  = int(abs(ycenter - (bbox_height/2)))
                        x_bottom_right = int(x_top_left + bbox_width)
                        y_bottom_right = int(y_top_left + bbox_height)
                        
                        #ground_truth_list.append([idx,int(label[0])+1,1,x_top_left,y_top_left,x_bottom_right,y_bottom_right])
                        img_gt.append((int(label[0])+1,1,x_top_left,y_top_left,x_bottom_right,y_bottom_right))
                ground_truth_dict.update({image_name:img_gt})

            t1 = time_sync()
            if img.shape[0]==1:
                img = np.squeeze(img, axis=0)
                img = torch.unsqueeze(torch.from_numpy(img), axis=0).to(device)
            else:
                img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img = img / 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            #visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(img, augment=augment, visualize=visualize)[0]
            
            t3 = time_sync()
            dt[1] += t3 - t2
            #################################################################
            if image_name not in inferences:
                inferences.update({image_name:[]})
            inferences[image_name].append(pred)
    #####################################################################################
    '''We need to preprocess the inferences dictionary to apply nms to every image with the 3 band inferences'''
    #####################################################################################
    # NMS
    for idx,img_name in enumerate(inferences):
        inferences[img_name] = torch.cat(inferences[img_name], dim=1)
        inferences[img_name] = non_max_suppression(inferences[img_name], conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        pred = inferences[img_name]
        for (gt_class,gt_conf,gt_x_top_left,gt_y_top_left,gt_x_bottom_right,gt_y_bottom_right) in ground_truth_dict[img_name]:
            ground_truth_list.append([idx,gt_class,gt_conf,gt_x_top_left,gt_y_top_left,gt_x_bottom_right,gt_y_bottom_right])
    #for idx in range(len(pred)):
        
        dt[2] += time_sync() - t3

        
        
        #gt_list = [([float(value)] for value in ]
        #gt_list = list(map(float, gt_list))

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            
            p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            s += '%gx%g ' % img.shape[2:]  # print string
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xyxy = torch.tensor(xyxy).view(1, 4).tolist()[0]
                    x_top_left,y_top_left,x_bottom_right,y_bottom_right = int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])
                    detections_list.append([idx,int(cls.item())+1,float(conf.item()),x_top_left,y_top_left,x_bottom_right,y_bottom_right])
    
    p,r,ap = eval_fh(detections_list, ground_truth_list, 0.4, 1)

    print('call to metrics our implementation')
    if p==0 and r==0:
        f1_result = 0
    else:  
        f1_result = (2.0 * p * r)/ (p + r)
    print()
    print("===============================================================")
    print("Precision:" + str(p))
    print("Recall:" + str(r))
    print("F1 Score:" + str(f1_result))
    print("===============================================================")
    with open(cvs_path, "a") as myfile:
        my_writer = csv.writer(myfile, delimiter=',', quotechar='"')
        if bands_to_apply:
            my_writer.writerow([bands_to_apply, len(ground_truth_list), len(detections_list), iou_thres, conf_thres, p, r, f1_result,weights])
        else:
            my_writer.writerow(["Visible light", len(ground_truth_list), len(detections_list), iou_thres, conf_thres, p, r, f1_result,weights])


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')#"path/to/Red/model/weights.pt path/to/Green/model/weights.pt path/to/Blue/model/weights.pt path/to/RedEdge/model/weights.pt path/to/NIR/model/weights.pt"
    parser.add_argument('--dataset_path', type=str, default=ROOT / 'data/', help='file/dir/to/dataset, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', type=boolean_string, default=False)
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    # Multispectral arguments
    parser.add_argument('--bands_to_apply', type=str, default="")#"Red Green Blue RedEdge NIR"
    # CSV file name to register results csv_file_name
    parser.add_argument('--csv_file_name', default='results_yolov5',type=str, help='csv file name to stores results')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
