# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 10:38:36 2021

@author: fabian

Simple Inference Script of EfficientDet-Pytorch
"""
import os
import time
import torch
from torch.backends import cudnn
from matplotlib import colors
from tqdm import tqdm

from backbone import EfficientDetBackbone
import cv2
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess_all, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box


force_input_size = None  # set None to use default size
use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True



class PyramidAnalysis:
    
    def __init__(self, weights_location, root_dir_testing, im_save):
        self.weights_location = weights_location
        self.root_dir_testing = root_dir_testing
        self.im_save = im_save
        
        if(not os.path.isdir(root_dir_testing + "_results/")):
            #os.system('mkdir', root_dir_testing + "_results/")
            os.makedirs(root_dir_testing + "_results/")
            
        self.destination_dir_images = root_dir_testing + '_results/'
        
        
        self.compound_coef = 4
        
        # replace this part with your project's anchor config
        self.anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
        self.anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
        
        self.threshold = 0.4
        self.iou_threshold = 0.4
        
        self.obj_list = ['pineapple']
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        self.input_size = self.input_sizes[self.compound_coef] if force_input_size is None else force_input_size
        
        #different colors for the pyramid's levels
        self.color_dict = {
                #level 3 -> red
                0:{'red':255, 'green':51, 'blue':0},
                
                #level 4 -> purple
                1:{'red':204, 'green':0, 'blue':255},
                
                #level 5 -> green
                2:{'red':0, 'green':204, 'blue':102},
                
                #level 6 -> light blue
                3:{'red':102, 'green':204, 'blue':255},
                
                #level 7 -> white
                4:{'red':255, 'green':255, 'blue':255}
                }
    
    
    
    #------------------------------------------------------------------------------
    def get_pyramid_color(self, pyramid_level):
        colors = self.color_dict[pyramid_level]
        return colors['red'], colors['green'], colors['blue']
    
    
    
    #------------------------------------------------------------------------------
    def display(self, preds, imgs, img_id_=0):
        
        for i in range(len(imgs)):
            if len(preds[i]['rois']) == 0:
                continue
    
            imgs[i] = imgs[i].copy()
    
            for j in range(len(preds[i]['rois'])):
                x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
                obj = self.obj_list[preds[i]['class_ids'][j]]
                score = float(preds[i]['scores'][j])
                
                #get a different color for every level of the pyramid
                red, green, blue = self.get_pyramid_color(preds[i]['pyramid_level'][j])
                #Coding for cv2: (Blue, Green, Red)
                my_color = (blue, green, red)
                plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj,score=score,color=my_color)
            
            if self.im_save:
                cv2.imwrite(self.destination_dir_images + "/" + str(img_id_) + ".jpg" , imgs[i])
            img_id_ += 1
                
        return img_id_
    
    
    
    #------------------------------------------------------------------------------
    def results_batch(self, images, img_id_):
        ori_imgs, framed_imgs, framed_metas = preprocess_all(images, max_size=self.input_size)
        
        if use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)
        
        x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)
        
        model = EfficientDetBackbone(compound_coef=self.compound_coef, 
                                     num_classes=len(self.obj_list),
                                     ratios=self.anchor_ratios, 
                                     scales=self.anchor_scales)

        model.load_state_dict(torch.load(self.weights_location, map_location='cpu'))#
        model.requires_grad_(False)
        model.eval()
        
        if use_cuda:
            model = model.cuda()
        if use_float16:
            model = model.half()
        
        with torch.no_grad():
            features, regression, classification, anchors = model(x)
        
            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()
        
            out, pyramid_count, _ = postprocess(x,
                                                anchors, 
                                                regression, 
                                                classification,
                                                regressBoxes, 
                                                clipBoxes,
                                                self.threshold, 
                                                self.iou_threshold,
                                                model.pyramid_limits)
        
        #
        out = invert_affine(framed_metas, out)
        new_id = self.display(out, ori_imgs, img_id_)
        
        return pyramid_count, new_id
    
    
    #------------------------------------------------------------------------------
    def get_batch(self, data, n=1):
        l = len(data)
        for ndx in range(0, l, n):
            yield data[ndx:min(ndx + n, l)]


'''
weights_location = 'weights/efficientdet-d4_5mts.pth'
root_dir_testing = "datasets/pineapple_215_8mts_train_val_test/test"
im_save = True
pyramid = PyramidAnalysis(weights_location, root_dir_testing, im_save)


original_names = os.listdir(root_dir_testing)
img_path = [root_dir_testing+'/'+i for i in original_names]

pyramid_acc = torch.zeros(5, dtype=torch.int32)
img_id = 0
for my_batch in pyramid.get_batch(img_path, 3):
    pyramid_temp, img_id = pyramid.results_batch(my_batch, img_id)
    pyramid_acc += pyramid_temp


print(pyramid_acc)'''


