import cv2
import numpy as np
import rasterio
from rasterio import plot
import matplotlib.pyplot as plt
from matplotlib import pyplot
import glob
import os
import shutil 

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
def get_band_combination(ms_imaga_np,combination_name='NDVI'):
    # Bands order at the raster file should be as ['Red','Green','Blue','RE','NIR']
    np.seterr(divide='ignore', invalid='ignore')
    red = ms_imaga_np[:,:,0]
    green = ms_imaga_np[:,:,1]
    blue = ms_imaga_np[:,:,2]
    re = ms_imaga_np[:,:,3]
    nir =ms_imaga_np[:,:,4]
    band_combinations ={
        'NDVI': (np.where((nir+red)==0., 0, np.divide((nir-red),(nir+red)))),
        'GNDVI': (np.where((nir+green)==0., 0, (nir-green)/(nir+green))),
        'NDRE': (np.where((nir+re+red)==0., 0, (nir-re)/(nir+red))),
        'LCI': (np.where((nir+re)==0., 0, (nir-re)/(nir+re))),
        'OSAVI': (np.where((nir+red)==0., 0, 1.6*((nir-red)/(nir-red+0.16))))
    }
    return band_combinations[combination_name]

# Read reference image
image_path = '/mnt/e/datosmanfred/gira_10_13_mar21_5m_sift_3_DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING_resized/00000068.JPG'
dirpath,_ = os.path.split(image_path)
image_path_no_ext,file_extension = os.path.splitext(image_path)
image_name = os.path.basename(image_path_no_ext)

colors = ['Red','Green','Blue','RedEdge','NIR']
bands = []
for band_color in colors:
    bands.append(cv2.imread(os.path.join(dirpath, f'{image_name}_{band_color}.TIF'),cv2.IMREAD_GRAYSCALE).astype('uint16'))
ms_image = np.dstack(bands)
print(ms_image.shape)
combination = get_band_combination(ms_image,'GNDVI')
fig = plt.figure(figsize=(12,12))
plot.show(combination)
#image = rasterio.open(image_path)
'''
image_path = '/mnt/e/datosmanfred/Gira 10 13 Mar21/Lote 71/5_meters/00000048.JPG'
dirpath,_ = os.path.split(image_path)
image_path_no_ext,file_extension = os.path.splitext(image_path)
image_name = os.path.basename(image_path_no_ext)

colors = ['Red','Green','Blue','RE','NIR']
bands = []
for band_color in colors:
    bands.append(rasterio.open(os.path.join(dirpath, f'{image_name}_{band_color}.TIF')).read(1).astype('float32'))
ms_image = np.dstack(bands)
print(ms_image.shape)
combination = get_band_combination(ms_image,'GNDVI')
fig = plt.figure(figsize=(12,12))
plot.show(combination)
#image = rasterio.open(image_path)'''