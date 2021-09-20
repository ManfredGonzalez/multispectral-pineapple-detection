import cv2
import numpy as np
import rasterio
from rasterio import plot
import matplotlib.pyplot as plt
from math import sqrt
import glob
import argparse
import shutil
from utils import getXMPData,boolean_string

MAX_FEATURES = 20000
GOOD_MATCH_PERCENT = 0.25
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

def align_dif_exposure_times(im1, im2,draw_matches=False, use_cuda=False):
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
    if not use_cuda:
        orb = cv2.ORB_create(MAX_FEATURES)
        keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
    else:
        orb = cv2.cuda_ORB_create(MAX_FEATURES)
        keypoints1, descriptors1 = orb.detectAndComputeAsync(im1Gray, None)
        keypoints2, descriptors2 = orb.detectAndComputeAsync(im2Gray, None)

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
    parser.add_argument('--use_cuda', type=boolean_string, default=False) 
    parser.add_argument('-fe', '--files_extension', type=str, default="JPG")

    args = parser.parse_args()
    return args

def preprocess_DJI_images(root_dir,ref_file_ext,results_path,use_cuda):
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
        use_cuda: (boolean) use gpu or not

    '''
    band_file_ext = 'TIF'
    myImages = glob.glob(f'{root_dir}/*.{ref_file_ext}')

    for ref_file_path in myImages:
        image_seed_name = ref_file_path[len(root_dir)+1:len(ref_file_path)-4]
        print("Reading reference image : ", ref_file_path)
        imReference = cv2.imread(ref_file_path, cv2.IMREAD_COLOR)
        # Shift the channels to RGB
        imReference = cv2.cvtColor(imReference, cv2.COLOR_BGR2RGB)
        imReference = cv2.cvtColor(imReference, cv2.COLOR_RGB2GRAY)

        shutil.copyfile(ref_file_path, f'{results_path}{image_seed_name}.{ref_file_ext}')
    
        bands = ['Red','Green','Blue','RE','NIR']
        ms_ref = rasterio.open(f"{root_dir}/{image_seed_name}_{bands[0]}.{band_file_ext}")
        rgb_array = []
        ms_image_all = rasterio.open(f'{results_path}{image_seed_name}.{band_file_ext}','w',driver='Gtiff',
                            width=ms_ref.width, 
                            height = ms_ref.height, 
                            count=5, crs=ms_ref.crs, 
                            transform=ms_ref.transform, 
                            dtype='float64')

        count = 1
        for band_name in bands:
            file_path = f"{root_dir}/{image_seed_name}_{band_name}.{band_file_ext}"
            ms_band = rasterio.open(file_path)
            meta_data = getXMPData(file_path)
            #band = ms_band.read(1)
            band = ms_band.read(1)
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
            
            
            band[band == 0] = 1

            #band[band == 0] = 1
            #band = normalize(ms_band.read(1),0,65535).astype('uint16')
            ms_image_all.write(band,count)
            count = count + 1
        
        ms_image_all.close()
    print('Done!!')

    
if __name__ == '__main__':

    opt = get_args()
    
    preprocess_DJI_images(opt.dir_path,opt.files_extension,opt.results_dir_path,opt.use_cuda)


