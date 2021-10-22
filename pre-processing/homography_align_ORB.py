import cv2
import numpy as np
import rasterio
from rasterio import plot
import matplotlib.pyplot as plt
from math import sqrt
import glob
import argparse
import shutil
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import getXMPData,boolean_string
from tqdm import tqdm

MAX_FEATURES = 50000
GOOD_MATCH_PERCENT = 0.40
def normalize(x, lower, upper):
    """ This is a simple linear normalization for an array to a given bound interval

        Params:
        x: image that needs to be normalized in this case
        lower: (int) the lower limit of the interval
        upper: (int) the upper limit of the interval

        Return:
        x_norm: the normalized image between the specified interval 
    """
    x_max = np.max(x)
    x_min = np.min(x)
    # The slope of the linear normalization
    m = (upper - lower) / (x_max - x_min)
    # Linear function for the normalization
    x_norm = (m * (x - x_min)) + lower

    return x_norm
def align_dif_camera_locations(img,x,y):
    '''
    Alignment of the phase difference caused by different camera locations.

    Params: 
    img: 2D numpy array that represents the image or the band that needs to be aligned
    x: (float) The  x direction  offset 
    y: (float) The  y direction  offset 

    Return:
    img: 2D numpy array that represents the aligned image
    '''
    transl_matrix = np.float32([[1, 0, x], [0, 1, y]])
    (rows, cols) = img.shape[:2]
    # warpAffine does appropriate shifting given the
    # translation matrix.
    res = cv2.warpAffine(img, transl_matrix, (cols, rows))
    return res

def align_dif_exposure_times(im1, im2,draw_matches=False):
    '''
    Alignment of the difference caused by different exposure times of the cameras. 
    A feature point detection is used to compute an alignment matrix (Homography) using several pairs of matched
    feature points.
    Params: 
    im1: 2D numpy array that represents the image or the band that needs to be aligned
    im2: 2D numpy array that represents the reference image to align the input image (im1)
    draw_matches:  (boolean) write the feature points that match between the images.
    use_cuda:  (boolean) use gpu or not

    Return:
    im1Reg: 2D numpy array that represents the aligned image
    h: 2D numpy array that represents the alignment matrix (Homography) used to align the input image.
    '''
    im1Gray = im1
    im2Gray = im2
    orb = cv2.ORB_create(MAX_FEATURES)
    # Detect ORB features and compute descriptors.
    
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
    

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    if draw_matches:
        imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
        cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h
def vignetting_corection(centerx,centery,K,picture):
    '''
     Vignetting correction model

     Params:
     centerx: (float) x coordinate of the vignette
     centery: (float) y coordinate of the vignette
     K: (list<int>) polynomial coefficients for Vignetting correction
     picture: 2D numpy array that represents the image or the band that needs to be corrected

     Return:
     correction: 2D numpy array that represents the corrected image

    '''
    height, width = picture.shape
    coefficient = np.zeros(picture.shape) 
    for y in range(1,height):
        for x in range(1,width):
            r = sqrt((x-centerx)**2 + (y-centery)**2)
            coefficient[y,x] = r**6*K[5]+r**5*K[4]+r**4*K[3]+r**3*K[2]+r**2*K[1]+r*K[0]+1
    correction = np.multiply(picture,coefficient)
    return correction
def pixel_correction(sensorgainadjustment, sensorgain, irradiance,exposuretime,picture):
    '''
    Pixel normalization and pixel correction

    Params:
    sensorgainadjustment: (float) This is the sensor gain adjustment of the camera
    sensorgain:  (float) the sensor gain setting (similar to the sensor ISO)
    irradiance: (float)  signal values relevant to the sunlight sensor
    exposuretime:  (int) the  camera exposure  time
    picture: 2D numpy array that represents the image or the band that needs to be corrected

    Return: 2D numpy array that represents the corrected image
    '''
    normalized_pixel = picture.astype('float64')/65535
    black_normalized=4096/65535
    substraction = normalized_pixel-black_normalized
    factor =sensorgain*(exposuretime/1e6)
    camera_pixel =substraction/factor
    factor2 = sensorgainadjustment/irradiance
    return camera_pixel*factor2
def get_args():
    """Get all expected parameters"""
    parser = argparse.ArgumentParser('EfficientDet Pytorch - Evaluate the model')
    parser.add_argument('-dp', '--dir_path', type=str, default="")
    parser.add_argument('-rd', '--results_dir_path', type=str, default="")
    parser.add_argument('--start_numbering', type=int, default=0)
    parser.add_argument('--use_cuda', type=boolean_string, default=False) 
    parser.add_argument('-fe', '--files_extension', type=str, default="JPG")
    parser.add_argument('--with_yv3_annot', type=boolean_string, default=False)
    parser.add_argument('--save_tif', type=boolean_string, default=False)

    args = parser.parse_args()
    return args
def numberName(number,totalDigits):
    additionalCeros = totalDigits-len(str(number))
    if additionalCeros < 0:
        return "number over the specified range"
    name = ""
    for digit in range(additionalCeros):
        name = "0" + name
    name = name + str(number)
    return name

def process(file_path,imReference):
    meta_data = getXMPData(file_path)
    #band = ms_band.read(1)
    band = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    vignettingData = [float(number) for number in meta_data['VignettingData'].split(',')]
    band = vignetting_corection(float(meta_data['CalibratedOpticalCenterX']),float(meta_data['CalibratedOpticalCenterY']),vignettingData,band)
    band = pixel_correction(float(meta_data['SensorGainAdjustment']), float(meta_data['SensorGain']), float(meta_data['Irradiance']),int(meta_data['ExposureTime']),band)
    band = normalize(band,0,255).astype('uint8')
    x = float(meta_data['RelativeOpticalCenterX'])
    y = float(meta_data['RelativeOpticalCenterY'])
    # Step 1
    band = align_dif_camera_locations(band,x,y)
    # Step 2
    # smooth the image (optional)
    #band = cv2.GaussianBlur(band,(5,5),0)
    band, homography = align_dif_exposure_times(band, imReference)
    return band

def preprocess_DJI_images(root_dir,ref_file_ext,results_path,start_numbering,with_yv3_annot=False,save_tif=False):
    '''
        This apply all the processing steps described at the Manual of the 
        drone DJI Phantom 4 Multispectral for image processing. Link of the manual:
        https://dl.djicdn.com/downloads/p4-multispectral/20200717/P4_Multispectral_Image_Processing_Guide_EN.pdf

        Images were first named with the script "namingScript.py" that stablishes an image seed name per 
        multispectral band of a single location shot of the drone.

        Params:
        root_dir: (string) Source path where images are located (ex: source/path/) 
        ref_file_ext: (string) Visible light images file extension (ex: JPG)
        results_path: (string) Directory where the results will be located
        with_yv3_annot: (boolean) organize the annotations if they exist

    '''
    band_file_ext = 'TIF'

    myImages = glob.glob(f'{root_dir}/*.{ref_file_ext}')
    myBands = glob.glob(f'{root_dir}/*.{band_file_ext}')
    images_dictionary = {}
    for rgb_image in myImages:
        meta_data = getXMPData(rgb_image)
        images_dictionary.update (
            {
                meta_data['CaptureUUID'] : [(rgb_image,'visible_light')]
            }
        )
    for tif_image in myBands:
        meta_data = getXMPData(tif_image)
        images_dictionary[meta_data['CaptureUUID']].append((tif_image,meta_data['BandName']))

    imagecounter = 0
    imagesNamesRange = range(start_numbering,100000000)  ####### Specify the starting number
    bands = ['Red','Green','Blue','RedEdge','NIR']
    for key in tqdm(images_dictionary, desc="Image pre-processing"):
        ref_file_path,band_name = images_dictionary[key][0]
        old_name = os.path.splitext(ref_file_path)[0]
        imageNewName = numberName(imagesNamesRange[imagecounter],8)
        imagecounter = imagecounter + 1
        imReference = cv2.imread(ref_file_path, cv2.IMREAD_COLOR)
        # Shift the channels to RGB
        imReference = cv2.cvtColor(imReference, cv2.COLOR_BGR2RGB)
        imReference = cv2.cvtColor(imReference, cv2.COLOR_RGB2GRAY)
        shutil.copyfile(ref_file_path, f'{results_path}{imageNewName}.{ref_file_ext}')
        if with_yv3_annot:
            shutil.copyfile(f'{old_name}.txt', f'{results_path}{imageNewName}.txt')

        tif_bands = images_dictionary[key][1:]
        nir_band = [item for item in tif_bands if item[1] == 'NIR'][0]
        tif_bands = [item for item in tif_bands if item[1] != 'NIR']
        for file_path,bandName in tif_bands:
        
            band = process(file_path,imReference)
            
            band[band == 0] = 1
            cv2.imwrite(f'{results_path}{imageNewName}_{bandName}.{band_file_ext}',band)
            
            if bandName == 'RedEdge':
                band = process(nir_band[0],band)
                band[band == 0] = 1
                cv2.imwrite(f'{results_path}{imageNewName}_{nir_band[1]}.{band_file_ext}',band)
    print('Done!!')


    
if __name__ == '__main__':

    opt = get_args()
    
    preprocess_DJI_images(opt.dir_path,opt.files_extension,opt.results_dir_path,opt.start_numbering, opt.with_yv3_annot,opt.save_tif)


