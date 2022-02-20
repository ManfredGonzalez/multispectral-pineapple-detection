# Author: Zylo117

import math
import os
import uuid
from glob import glob
from typing import Union
import numpy as np
import cv2
np.seterr(divide='ignore', invalid='ignore')
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
        ### The Basic Vegetation Indices
        'RVI': (np.where((nir+red)==0., 0, red/nir)+1)*127.5,
        'DVI': (np.where((nir+red)==0., 0, nir-red)+1)*127.5,
        'VDVI': (np.where((red+green+blue)==0., 0, ((2*green)-red-blue)/((2*green)+red+blue))+1)*127.5,### just RGB
        'ExGI': (np.where((red+green+blue)==0., 0, (2*green)-(red+blue))+1)*127.5, ### just RGB
        'GCC': (np.where((red+green+blue)==0., 0, green/(red+green+blue))+1)*127.5,### just RGB
        ### Indices that address Atmospheric (and other) Effects
        'EVI': (np.where((nir+red+blue)==0., 0, 2.5*((nir-red)/(nir+(6*red)-(7.5*blue)+1)))+1)*127.5, # NOT VISUAL
        'ARVI': (np.where((nir+red+blue)==0., 0, (nir-(red*blue))/(nir+(red*blue)))+1)*127.5,
        'GARI': (np.where((nir+green+red+blue)==0., 0, (nir-(green-(1.7*(blue-red))))/(nir+(green-(1.7*(blue-red)))))+1)*127.5,
        'VARI': (np.where((red+green+blue)==0., 0, (green-red)/(green+red-blue))+1)*127.5,### just RGB
        ### Addressing Soil Reflectance
        #SAVI requires a priori information about vegetation presence in the study area. We do not have that
        #MSAVI a modified SAVI (MSAVI) replaces 𝐿 factor in the SAVI equation (8) with a variable 𝐿 function. Based on the implementation of the MSAVI, Richardson
            #and Wiegand (1977) proposed a Modified Secondary Soil-
            #Adjusted Vegetation Index (MSAVI2)(Xue et al. 2017)
        'MSAVI12': (np.where((nir+red)==0., 0, (0.5*((2*nir+1)-np.sqrt(np.power((2*nir+1),2)-(8*(nir-red))))))+1)*127.5,
        'MCARI': (np.where((red+re+green)==0., 0, ((re-red)-0.2*(re-green)*(re/red)))+1)*127.5,
        'SIPI': (np.where((nir+blue+red)==0., 0, ((nir-blue)/(nir-red)))+1)*127.5, # NOT VISUAL
        #Agricultural Indices
        'TGI': (np.where((red+green+blue)==0., 0, 0.5*(((650-450)*(red-green))-((650-560)*(red-blue))))+1)*127.5,### just RGB
        'GLI': (np.where((red+green+blue)==0., 0, (((green-red)+(green-blue))/((2*green)+(blue+red))))+1)*127.5,### just RGB
        #Task-specific Vegetation Indices
        #(TDVI) was developed to detect vegetation in urban settings where NDVI is often saturated. 
        'TDVI': (np.where((nir+red)==0., 0, 1.5*(nir-red)/np.sqrt(np.power(nir,2)+red+0.5))+1)*127.5,
        'NDVI': (np.where((nir+red)==0., 0, np.divide((nir-red),(nir+red)))+1)*127.5,
        'GNDVI': (np.where((nir+green)==0., 0, (nir-green)/(nir+green))+1)*127.5,
        'NDRE': (np.where((nir+re)==0., 0, (nir-re)/(nir+re))+1)*127.5,
        'LCI': (np.where((nir+re+red)==0., 0, (nir-re)/(nir+red))+1)*127.5,
    }
    return band_combinations[combination_name]
def is_vegetation_index(bandName):
    indexes = ['RVI','DVI','VDVI','ExGI','GCC','ARVI','GARI','VARI','MSAVI12','MCARI','TGI','GLI','TDVI','NDVI','GNDVI','NDRE','LCI']
    return bandName in indexes