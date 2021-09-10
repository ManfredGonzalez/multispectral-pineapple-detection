import cv2
import numpy as np
import rasterio
from rasterio import plot
import matplotlib.pyplot as plt
from math import sqrt
from utils import getXMPData

MAX_FEATURES = 20000
GOOD_MATCH_PERCENT = 0.25
# This is a simple linear normalization
def normalize(x, lower, upper):
    """Normalize an array to a given bound interval"""
    x_max = np.max(x)
    x_min = np.min(x)
    # The slope of the linear normalization
    m = (upper - lower) / (x_max - x_min)
    # Linear function for the normalization
    x_norm = (m * (x - x_min)) + lower

    return x_norm
def align_dif_camera_locations(img,x,y):
    transl_matrix = np.float32([[1, 0, x], [0, 1, y]])
    (rows, cols) = img.shape[:2]
    # warpAffine does appropriate shifting given the
    # translation matrix.
    res = cv2.warpAffine(img, transl_matrix, (cols, rows))
    return res

def align_dif_exposure_times(im1, im2,draw_matches=False):
    #im1Gray = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    #im2Gray = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
    im1Gray = im1
    im2Gray = im2
    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
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
def save_new_band(path,reference_band,matrix):
    new_band = rasterio.open(path,'w',driver='Gtiff',
                          width=reference_band.width, 
                          height = reference_band.height, 
                          count=1, crs=reference_band.crs, 
                          transform=reference_band.transform, 
                          dtype='float64')
    new_band.write(matrix,1)
    new_band.close()
def get_band_combination(dataset_reader,combination_name='NDVI'):
    # Bands order at the raster file should be as ['Red','Green','Blue','RE','NIR']
    red = dataset_reader.read(1)
    green = dataset_reader.read(2)
    blue = dataset_reader.read(3)
    re = dataset_reader.read(4)
    nir = dataset_reader.read(5)
    combinations = []
    combinations.append(normalize(np.where((nir+red)==0., 0, np.divide((nir-red),(nir+red))),0,255).astype('uint8'))
    '''combinations.append((np.where((nir+green)==0., 0, (nir-green)/(nir+green))))
    combinations.append((np.where((nir+re+red)==0., 0, (nir-re)/(nir+red))))
    combinations.append((np.where((nir+re)==0., 0, (nir-re)/(nir+re))))
    combinations.append((np.where((nir+red)==0., 0, 1.6*((nir-red)/(nir-red+0.16)))))'''
    band_combinations ={
        'NDVI': 0,
        'GNDVI': 1,
        'NDRE': 2,
        'LCI': 3,
        'OSAVI': 4
    }
    return combinations[0]
def vignetting_corection(centerx,centery,K,picture):
    height, width = picture.shape
    coefficient = np.zeros(picture.shape) 
    for y in range(1,height):
        for x in range(1,width):
            r = sqrt((x-centerx)**2 + (y-centery)**2)
            coefficient[y,x] = r**6*K[5]+r**5*K[4]+r**4*K[3]+r**3*K[2]+r**2*K[1]+r*K[0]+1
    correction = np.multiply(picture,coefficient)
    return correction
def pixel_correction(sensorgainadjustment, sensorgain, irradiance,exposuretime,picture):
    normalized_pixel = picture.astype('float64')/65535
    black_normalized=4096/65535
    substraction = normalized_pixel-black_normalized
    factor =sensorgain*(exposuretime/1e6)
    camera_pixel =substraction/factor
    factor2 = sensorgainadjustment/irradiance
    return camera_pixel*factor2
if __name__ == '__main__':
    # Read reference image
    root_dir = 'data/multispectral_image'
    image_seed_name = '00000022'
    ref_file_ext = 'jpg'
    band_file_ext = 'TIF'
    ref_file_path = f"{root_dir}/{image_seed_name}.{ref_file_ext}"
    results_path = 'results/'

    camera_locations = {
        'red':(-0.12500,-0.25000),'green':(-1.12500,3.93750),
        'blue':(4.15625,2.18750),'re':(1.31250,4.15625),'nir':(0.00000,0.00000)
    }

    print("Reading reference image : ", ref_file_path)
    imReference = cv2.imread(ref_file_path, cv2.IMREAD_COLOR)
    # Shift the channels to RGB
    imReference = cv2.cvtColor(imReference, cv2.COLOR_BGR2RGB)
    imReference = cv2.cvtColor(imReference, cv2.COLOR_RGB2GRAY)

  
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
        meta_data = getXMPData('data/multispectral_image/00000022_NIR.TIF')
        #band = ms_band.read(1)
        band = ms_band.read(1)
        vignettingData = [float(number) for number in meta_data['VignettingData'].split(',')]
        band = vignetting_corection(float(meta_data['CalibratedOpticalCenterX']),float(meta_data['CalibratedOpticalCenterY']),vignettingData,band)
        band = pixel_correction(float(meta_data['SensorGainAdjustment']), float(meta_data['SensorGain']), float(meta_data['Irradiance']),int(meta_data['ExposureTime']),band)
        band = normalize(band,0,255).astype('uint8')
        x,y = camera_locations[band_name.lower()]
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

    #raster = rasterio.open(f'{results_path}{image_seed_name}.{band_file_ext}')
    #prueba = raster.read(1)
    #combination = get_band_combination(raster,combination_name='NDVI')
    #print(combination)

