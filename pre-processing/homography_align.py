import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import glob
import argparse
import shutil
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import getXMPData,boolean_string
from tqdm import tqdm
from numba import jit
import time
from mgh_extractor import Mgh_feature_extractor

import rasterio
from scipy import ndimage

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

def align_dif_exposure_times_ECC(im1_gray, im2_gray):
    #im_to_be_alligned = im2_gray
    #im1_gray = ndimage.sobel(im1_gray)
    #im2_gray = ndimage.sobel(im2_gray)
    # Find size of image1
    sz = im1_gray.shape

    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 5000

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)

    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective (im2_gray, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2_gray, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return im2_aligned

def align_dif_exposure_times(im1, im2,draw_matches=False,feature_extractor='ORB',matcher='DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING'):
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

    
    # Detect ORB features and compute descriptors.
    if feature_extractor=='SIFT':
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(im1Gray, None)
        keypoints2, descriptors2 = sift.detectAndCompute(im2Gray, None)
    elif feature_extractor=='ORB':
        orb = cv2.ORB_create(50000)
        keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
    elif feature_extractor=='mgh':
        keypoints1, descriptors1 = Mgh_feature_extractor(im1Gray).getFeatures()
        keypoints2, descriptors2 = Mgh_feature_extractor(im2Gray).getFeatures()
    else:
        raise Exception(f'FEATURE EXTRACTOR NAME ERROR: Images can only be alligned with the algorithms named SIFT,ORB or mgh')



    # Match features.
    if matcher=='DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING':
        bf = cv2.BFMatcher(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING, crossCheck=True)
    elif matcher=='NORM_L1':
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    else:
        raise Exception(f'FEATURE MATCHER NAME ERROR: Images can only be alligned with the matcher named DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING or NORM_L1')

    matches = bf.match(descriptors1,descriptors2)
    

    matches = sorted(matches, key = lambda x:x.distance)

    # Remove not so good matches
    #numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    #matches = matches[:numGoodMatches]

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

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
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
    picture = picture.astype('float64')
    normalized_pixel = picture/65535
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
    parser.add_argument('--feature_extractor', type=str, default="SIFT")# it could be ORB, SIFT, or mgh, ECC
    parser.add_argument('--matcher', type=str, default="NORM_L1")# also it could be DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING or NORM_L1


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

def process(file_path,imReference,feature_extractor,matcher):
    meta_data = getXMPData(file_path)
    #band = ms_band.read(1)
    #band = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    band = rasterio.open(file_path).read(1)
    vignettingData = [float(number) for number in meta_data['VignettingData'].split(',')]
    band = vignetting_corection(float(meta_data['CalibratedOpticalCenterX']),float(meta_data['CalibratedOpticalCenterY']),vignettingData,band)
    band = pixel_correction(float(meta_data['SensorGainAdjustment']), float(meta_data['SensorGain']), float(meta_data['Irradiance']),int(meta_data['ExposureTime']),band)
    band = normalize(band,0,127).astype('uint8')
    x_percentage = float(meta_data['RelativeOpticalCenterX'])
    y_percentage = float(meta_data['RelativeOpticalCenterY'])
    '''height,width = band.shape
    x = width * (x_percentage/100)
    y = height * (y_percentage/100)
    # Step 1
    band = align_dif_camera_locations(band,x,y)'''
    # Step 2
    # smooth the image (optional)
    #band = cv2.GaussianBlur(band,(5,5),0)
    if feature_extractor=='ECC':
        band = align_dif_exposure_times_ECC(imReference, band)
    else: 
        band, homography = align_dif_exposure_times(band, imReference,feature_extractor=feature_extractor,matcher=matcher)
    
    return band

def preprocess_DJI_images(root_dir,ref_file_ext,results_path,start_numbering,feature_extractor,matcher,with_yv3_annot=False,save_tif=False):
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
    if not os.path.exists(results_path):
        os.makedirs(results_path)
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
    for key in tqdm(images_dictionary, desc="Image pre-processing"):
        ref_file_path,band_name = images_dictionary[key][0]
        #ref_file_path,band_name = [item for item in images_dictionary[key][1:] if item[1] == 'NIR'][0]
        old_name = os.path.splitext(ref_file_path)[0]
        imageNewName = numberName(imagesNamesRange[imagecounter],8)
        imagecounter = imagecounter + 1
        #imReference = cv2.imread(ref_file_path, cv2.IMREAD_GRAYSCALE)
        imReference = cv2.imread(ref_file_path, cv2.IMREAD_COLOR)
        # Shift the channels to RGB
        imReference = cv2.cvtColor(imReference, cv2.COLOR_BGR2RGB)
        imReference = cv2.cvtColor(imReference, cv2.COLOR_RGB2GRAY)
        shutil.copyfile(ref_file_path, f'{results_path}/{imageNewName}.{ref_file_ext}')
        if with_yv3_annot:
            shutil.copyfile(f'{old_name}.txt', f'{results_path}/{imageNewName}.txt')

        tif_bands = images_dictionary[key][1:]
        nir_band = [item for item in tif_bands if item[1] == 'NIR'][0]
        re_band = [item for item in tif_bands if item[1] == 'RedEdge'][0]
        red_band = [item for item in tif_bands if item[1] == 'Red'][0]
        if feature_extractor=='ECC':
            tif_bands = [item for item in tif_bands if item[1] != 'NIR']
            tif_bands = [item for item in tif_bands if item[1] != 'RedEdge']
            tif_bands = [item for item in tif_bands if item[1] != 'Red']

        imReference = normalize(imReference,0,127).astype('uint8')
        for file_path,bandName in tif_bands:
        
            band = process(file_path,imReference,feature_extractor,matcher)
            
            band[band == 0] = 1
            cv2.imwrite(f'{results_path}/{imageNewName}_{bandName}.{band_file_ext}',band)
            '''if bandName=='Blue' and feature_extractor=='ECC':
                #Blue aligns Red band
                ref_band = band'''
                

            if bandName=='Green' and feature_extractor=='ECC':
                #Green aligns RedEdge band
                ref_band = band
                band = process(red_band[0],ref_band,feature_extractor,matcher)
                band[band == 0] = 1
                cv2.imwrite(f'{results_path}/{imageNewName}_{red_band[1]}.{band_file_ext}',band)
                band1 = process(re_band[0],ref_band,feature_extractor,matcher)
                band1[band1 == 0] = 1
                cv2.imwrite(f'{results_path}/{imageNewName}_{re_band[1]}.{band_file_ext}',band1)
                #RedEdge aligns NIR
                ref_band = band1
                band = process(nir_band[0],ref_band,feature_extractor,matcher)
                band[band == 0] = 1
                cv2.imwrite(f'{results_path}/{imageNewName}_{nir_band[1]}.{band_file_ext}',band)
            
    print('Done!!')


    
if __name__ == '__main__':

    opt = get_args()
    start = time.time()
    preprocess_DJI_images(opt.dir_path,opt.files_extension,opt.results_dir_path,opt.start_numbering,opt.feature_extractor,opt.matcher,opt.with_yv3_annot,opt.save_tif)
    end = time.time()
    print("Elapsed (after compilation) = %s" % (end - start))


