# Author: Zylo117

import math
import os
import uuid
from glob import glob
from typing import Union
import numpy as np
import cv2

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def get_band_combination(ms_imaga_np,combination_name='NDVI'):
    # Bands order at the raster file should be as ['Red','Green','Blue','RE','NIR']
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
def is_vegetation_index(bands_to_apply):
    return 'NDVI' in bands_to_apply or 'GNDVI' in bands_to_apply or 'NDRE' in bands_to_apply or 'LCI' in bands_to_apply or 'OSAVI' in bands_to_apply