
import os 
from shutil import copyfile
from sklearn.model_selection import train_test_split
import numpy as np
import random
import shutil
import json
import cv2
import yaml
import glob
from utils.utils import boolean_string
import argparse
import sys

#sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def copy_data():
    """
    In case I have to copy the files into the root file of the dataset folder

    Returns
    -------
    None.

    """
    #get all dir content
    file_input_dir = 'dataset2/'
    names = os.listdir(file_input_dir)
    
    for n in names:
        
        folder_name = file_input_dir + n
        files = os.listdir(folder_name)
        for f in files:
            file_name = f.split('.')[0]
            file_extension = f.split('.')[1]
            if(file_extension == "txt"):
                copyfile(folder_name + "/" + file_name + ".jpg", file_input_dir + file_name + '.jpg')
                copyfile(folder_name + "/" + file_name + ".txt", file_input_dir + file_name + '.txt')
                copyfile(folder_name + "/" + file_name + ".xml", file_input_dir + file_name + '.xml')
    


    
def split_data(file_input_dir, output_folder, 
                classes_file, 
                name_1, name_2, name_3, 
                set_1, set_2, set_3, 
                shuffle, sub_sample, seed, img_extension,
                multispectral=True, 
                annotations_file=None):
    """
    Method to split into train/test/val sets.

    Parameters
    :ratio_train_test(float) first split between train and another chunck (ratio).
    :ratio_test_val(float) ratio split of the chunck data into test and valid.
    :shuffle(boolean) indicates if we have to pick a subset of random data.
    :sub_sample(int) number of random samples to be selected.

    Returns
    :None.
    """
    #get data
    if annotations_file:
        anns_names_list, anns_bboxes_list, class_list = files_to_array(file_input_dir + annotations_file, 
                                                                       file_input_dir + classes_file)
    else:
        anns_names_list, anns_bboxes_list, class_list = files_to_array_yolov3(file_input_dir, 
                                                                              file_input_dir + classes_file,
                                                                              img_extension)
    #check if the splits make sense
    if set_1 + set_2 + set_3 != 1.0:
        raise Exception("The split should sum 1.0")
    
    #--------------------------------------------------------------------------
    #1. Delete folders
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    '''
    if os.path.exists(output_folder + name_1 + "/"):
        shutil.rmtree(output_folder + name_1 + "/")
    if os.path.exists(output_folder + name_2 + "/"):
        shutil.rmtree(output_folder + name_2 + "/")
    if os.path.exists(output_folder + name_3 + "/"):
        shutil.rmtree(output_folder + name_3 + "/")
    if os.path.exists(output_folder + "annotations" + "/"):
        shutil.rmtree(output_folder + "annotations" + "/")
    '''
    while os.path.exists(output_folder):
        print("waiting")
        pass
    
    #2. Create directories
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if not os.path.exists(output_folder + name_1 + "/"):
        os.mkdir(output_folder + name_1 + "/")
    if not os.path.exists(output_folder + name_2 + "/"):
        os.mkdir(output_folder + name_2 + "/")
    if not os.path.exists(output_folder + name_3 + "/") and (set_3 != 0.0):
        os.mkdir(output_folder + name_3 + "/")
    if not os.path.exists(output_folder + "annotations" + "/"):
        os.mkdir(output_folder + "annotations" + "/")
    #--------------------------------------------------------------------------
    
    
    #--------------------------------------------------------------------------
    #3. Get image names and keep them into an array
    names = os.listdir(file_input_dir)
        
    #CODE TO SPLIT FILES INTO FOLDER
    #get file names without extension
    names_ = []
    for name in names:
        file_extension = name[-3:]
        file_name = name[:-4]
    
        if file_extension.upper() == img_extension.upper():        
            names_.append(file_name.strip())
    
    #If we want just a subset
    if(sub_sample > 0):
        random.shuffle(names_)
        names_ = names_[:sub_sample]
    #--------------------------------------------------------------------------
    
    
    #------------------------------------------------------------------------------
    #4. Split data and copy files
    y_dumpy = np.zeros(len(names_))
    X_train, X_test_big, _, y_test_big = train_test_split(names_, y_dumpy, train_size = set_1, shuffle=shuffle, random_state=seed)
    
    
    #Copy into the training
    #-------------------
    print(f"{name_1}: ", len(X_train))
    json_data_train = lists_to_json(class_list)
    
    ann_counter = 0
    for img_index, img_name in enumerate(X_train):    
        folder = output_folder + name_1 + "/"
        copyfile(file_input_dir + img_name + '.' + img_extension, folder + img_name + '.' + img_extension)
        if multispectral:
            copyfile(file_input_dir + img_name + '.TIF', folder + img_name + '.TIF')
        #include annotations in the json
        final_json_train, ann_counter = annotations_to_json(file_input_dir, img_name, img_index, img_extension, ann_counter,
                                                                  anns_names_list, anns_bboxes_list, json_data_train)

        #save json file
        with open(output_folder + "annotations/" + "instances_" + name_1 +'.json', 'w') as outfile:
            json.dump(final_json_train, outfile, indent=4)
    #-------------------
    
    
    #Use two or three sets for splitting
    if(set_3 != 0.0):
        new_val = set_2 / (set_2 + set_3)
        X_val, X_test, _, _ = train_test_split(X_test_big, y_test_big, train_size = new_val, shuffle=shuffle, random_state=seed)
        
        #Copy into the val and create json
        #-------------------
        print(f"{name_2}: ", len(X_val))
        json_data_val = lists_to_json(class_list)
        
        ann_counter = 0
        for img_index, img_name in enumerate(X_val):        
            folder = output_folder + f"{name_2}/"
            copyfile(file_input_dir + img_name + '.' + img_extension, folder + img_name + '.' + img_extension)
            if multispectral:
                copyfile(file_input_dir + img_name + '.TIF', folder + img_name + '.TIF')
            #include annotations in the json
            final_json_val, ann_counter = annotations_to_json(file_input_dir, img_name, img_index, img_extension, ann_counter,
                                                                      anns_names_list, anns_bboxes_list, json_data_val)
            
            #save json file
            with open(output_folder + "annotations/" + "instances_" + name_2 +'.json', 'w') as outfile:
                json.dump(final_json_val, outfile, indent=4)
        #-------------------
        
        
        #Copy into the test and create json
        #-------------------
        print(f"{name_3}: ", len(X_test))
        json_data_test = lists_to_json(class_list)
        
        ann_counter = 0
        for img_index, img_name in enumerate(X_test):   
            #move image
            folder = output_folder + f"{name_3}/"
            copyfile(file_input_dir + img_name + '.' + img_extension, folder + img_name + '.' + img_extension)
            if multispectral:
                copyfile(file_input_dir + img_name + '.TIF', folder + img_name + '.TIF')
            #include annotations in the json
            final_json_test, ann_counter = annotations_to_json(file_input_dir, img_name, img_index, img_extension, ann_counter,
                                                                      anns_names_list, anns_bboxes_list, json_data_test)
            
            #save json file
            with open(output_folder + "annotations/" + "instances_" + name_3 +'.json', 'w') as outfile:
                json.dump(final_json_test, outfile, indent=4)
        #-------------------
        
    else:
        #Copy into the testing
        #-------------------
        print(f"{name_2}: ", len(X_test_big))
        json_data_test = lists_to_json(class_list)
        
        ann_counter = 0
        for img_index, img_name in enumerate(X_test_big):        
            folder = output_folder + f"{name_2}/"
            copyfile(file_input_dir + img_name + '.' + img_extension, folder + img_name + '.' + img_extension)
            if multispectral:
                copyfile(file_input_dir + img_name + '.TIF', folder + img_name + '.TIF')
            #include annotations in the json
            final_json_test, ann_counter = annotations_to_json(file_input_dir, img_name, img_index, img_extension, ann_counter,
                                                                      anns_names_list, anns_bboxes_list, json_data_test)
            
            #save json file
            with open(output_folder + "annotations/" + "instances_" + name_2 +'.json', 'w') as outfile:
                json.dump(final_json_test, outfile, indent=4)
        #-------------------
    return class_list
    
def files_to_array(annotations_file, class_file):
    """
    Reads files of annotations and classes and returns the info into lists.

    Params
    :annotations_file(str) path to annotations (format: yolo v4).
    :class_file(str) path to annotations (format: yolo v4).

    Returns
    :annotaions_result(list of tuples) format: [('image_name.jpg', [anno_1, anno_2]), ... , (...)]
    :class_file(list of classes) string list of the names of the classes.
    """
    with open(annotations_file, "r") as my_file:
        annotations_list = my_file.read().split('\n')
    with open(class_file, "r") as my_file:
        #class_list = my_file.read().split('\n')
        for line in my_file.read().split('\n'):
            if len(line.strip()) > 1:
                class_list.append(line)
    
    annotations_names = []
    annotations_bboxes = []
    for annotation_line in annotations_list:
        array = annotation_line.split(' ')
        annotations_names.append(array[0])
        annotations_bboxes.append(array[1:])
    return annotations_names, annotations_bboxes, class_list

def files_to_array_yolov3(file_input_dir, class_file,img_extension):
    """
    Read files of annotations (yolo v3) and classes and returns the info into lists.

    Params
    :annotations_directory(str) path to annotations (format: yolo v3).
    :class_file(str) path to annotations (format: yolo v3).

    Returns
    :annotaions_result(list of tuples) format: [('image_name.jpg', [anno_1, anno_2]), ... , (...)]
    :class_file(list of classes) string list of the names of the classes.
    """
    class_list = []
    with open(class_file, "r") as my_file:
        #class_list = my_file.read().split('\n')
        for line in my_file.read().split('\n'):
            if len(line.strip()) > 1:
                class_list.append(line)

    myImages = [f for f in os.listdir(file_input_dir) if f.endswith(f'.{img_extension}')]
    
    annotations_names = []
    annotations_bboxes = []
    for file_image_name in myImages:
        img_path = file_input_dir + file_image_name
        imgRGB = cv2.imread(img_path)
        imageHeight, imageWidth, channels = imgRGB.shape
        yoloFile = open(img_path[:len(img_path)-4]+'.txt','r').readlines()
        
        annotations = []
        for label in yoloFile:
            label = label.split()
            xcenter = float(label[1]) * imageWidth
            ycenter = float(label[2]) * imageHeight
            bbox_width = float(label[3]) * imageWidth
            bbox_height = float(label[4]) * imageHeight
            x_top_left = int(abs(xcenter-(bbox_width/2)))
            y_top_left  = int(abs(ycenter - (bbox_height/2)))
            x_bottom_right = x_top_left + bbox_width
            y_bottom_right = y_top_left + bbox_height
            annotations.append(f'{x_top_left},{y_top_left},{x_bottom_right},{y_bottom_right},{label[0]}')
        annotations_names.append(file_image_name)
        annotations_bboxes.append(annotations)
    return annotations_names, annotations_bboxes, class_list

def lists_to_json(class_list):
    """
    Transform the class list into json.

    Params
    :class_file(list of classes) string list of the names of the classes.

    Returns
    :anns_list(json) correct format of json as defined in coco.
    """
    json_data = {}
    #print(annotation_array)
    
    #main nodes
    json_data['info'] = {}
    json_data['licenses'] = []
    json_data['categories'] = []
    json_data['images'] = []
    json_data['annotations'] = []

    #license, not used but included
    json_data['info']['year'] = "2021"
    json_data['info']['version'] = "11"
    json_data['info']['description'] = ""
    json_data['info']['contributor'] = ""
    json_data['info']['url'] = ""
    json_data['info']['date_created'] = "2041-06-16T14:01:38+00:00"
    
    json_data['licenses'].append({
        'id': 1,
        'url': '',
        'name': 'Unknown'
    })
    
    #categories the id will be the position in the array
    for i, item in enumerate(class_list):
        json_data['categories'].append({
            'id': i+1,
            'name': item,
            'supercategory': 'none'
        })
        
    #images and annotation should be included later on
    
    return json_data

def annotations_to_json(file_input_dir, img_name, img_index, img_extension, 
                        ann_counter, anns_names_list, anns_bboxes_list, json_data):
    #get width and height
    im = cv2.imread(file_input_dir + img_name + '.' + img_extension)
    h, w, c = im.shape

    json_data['images'].append({
        'id': img_index,
        'license': 1,
        'file_name': img_name + '.' + img_extension,
        'height': h,
        'width': w,
        'date_captured': '2041-03-03T03:54:32+00:00'

    })

    #get annotations of this image
    annotation_index = anns_names_list.index(img_name + '.' + img_extension)
    for annotation in anns_bboxes_list[annotation_index]:
        x1,y1,x2,y2,class_id = annotation.split(',')
        x1,y1,x2,y2,class_id = float(x1), float(y1), float(x2), float(y2), int(class_id)
        
        json_data['annotations'].append({
            'id': ann_counter,
            'image_id': img_index,
            'category_id': class_id + 1,
            'bbox': [x1, y1, x2 - x1, y2 - y1],
            'area': (x2 - x1) * (y2 - y1),
            'segmentation': [],
            'iscrowd': 0
        })
        ann_counter += 1
    return json_data, ann_counter


def create_project_file(project_name, output_project_file, train_, val_, test_, unlabeled_, obj_list):
    with open(output_project_file, 'w') as my_file:
        my_file.write(f'project_name: {project_name}\n')
        my_file.write(f'train_set: {train_}\n')
        my_file.write(f'train_set_unlabeled: {unlabeled_}\n')
        my_file.write(f'val_set: {val_}\n')
        my_file.write(f'test_set: {test_}\n')
        my_file.write('num_gpus: 1\n')
        my_file.write('mean: [0.485, 0.456, 0.406]\n')
        my_file.write('std: [0.229, 0.224, 0.225]\n')
        my_file.write("anchors_scales: '[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]'\n")
        my_file.write("anchors_ratios: '[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]'\n")
        my_file.write(f'obj_list: {obj_list}\n')


def get_args():
    """Get all expected parameters"""
    parser = argparse.ArgumentParser('Create Yolo v3-v4 datasets to COCO')
    parser.add_argument('--yolo_version', type=str, default='v3')
    parser.add_argument('--input_path', type=str, default='datasets/yolo_format/apple_yolov4pytorch/')
    parser.add_argument('--annotations_file', type=str, default='_annotations.txt')
    parser.add_argument('--classes_file', type=str, default='classes.txt')
    parser.add_argument('-p', '--project_name', type=str, default='5m_pineapple_10')
    parser.add_argument('--set_1', type=str, default='train')
    parser.add_argument('--set_2', type=str, default='val')
    parser.add_argument('--set_3', type=str, default='test')
    parser.add_argument('--set_4', type=str, default='None')
    parser.add_argument('--ratio_set_1', type=float, default=0.7) 
    parser.add_argument('--ratio_set_2', type=float, default=0.15) 
    parser.add_argument('--ratio_set_3', type=float, default=0.15)
    parser.add_argument('--shuffle', type=boolean_string, default=True) 
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--sub_sample', type=int, default=0)
    parser.add_argument('--img_extension', type=str, default='jpg')
    parser.add_argument('--multispectral', type=boolean_string, default=True) 

    args = parser.parse_args()
    return args
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    opt = get_args()
    
    output_folder = 'datasets/' + opt.project_name + '/'
    output_yml = 'projects/' + opt.project_name + '.yml'

    #run split
    if opt.yolo_version == 'v3':
        class_list = split_data(opt.input_path, output_folder, opt.classes_file, 
                                opt.set_1, opt.set_2, opt.set_3,
                                opt.ratio_set_1, opt.ratio_set_2, opt.ratio_set_3, 
                                opt.shuffle, opt.sub_sample, opt.seed, opt.img_extension,
                                multispectral = opt.multispectral,annotations_file = None)
                                
        create_project_file(opt.project_name, output_yml, opt.set_1, opt.set_2, opt.set_3, opt.set_4, class_list)

    elif opt.yolo_version == 'v4': 
        class_list = split_data(opt.input_path, output_folder, opt.classes_file, 
                                opt.set_1, opt.set_2, opt.set_3,
                                opt.ratio_set_1, opt.ratio_set_2, opt.ratio_set_3, 
                                opt.shuffle, opt.sub_sample, opt.seed, opt.img_extension,
                                multispectral = opt.multispectral,annotations_file = opt.annotations_file)
        create_project_file(opt.project_name, output_yml, opt.set_1, opt.set_2, opt.set_3, opt.set_4, class_list)
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))