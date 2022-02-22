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
    # Bands order at the raster file should be as ['ms_imaga_np[:,:,0]','ms_imaga_np[:,:,1]','ms_imaga_np[:,:,2]','RE','ms_imaga_np[:,:,4]']
        ### The Basic Vegetation Indices
    if combination_name=='RVI':
        return (np.where((ms_imaga_np[:,:,4]+ms_imaga_np[:,:,0])==0., 0, ms_imaga_np[:,:,0]/ms_imaga_np[:,:,4])+1)*127.5
    elif combination_name=='DVI':
        return (np.where((ms_imaga_np[:,:,4]+ms_imaga_np[:,:,0])==0., 0, ms_imaga_np[:,:,4]-ms_imaga_np[:,:,0])+1)*127.5
    elif combination_name=='VDVI':
        return (np.where((ms_imaga_np[:,:,0]+ms_imaga_np[:,:,1]+ms_imaga_np[:,:,2])==0., 0, ((2*ms_imaga_np[:,:,1])-ms_imaga_np[:,:,0]-ms_imaga_np[:,:,2])/((2*ms_imaga_np[:,:,1])+ms_imaga_np[:,:,0]+ms_imaga_np[:,:,2]))+1)*127.5 ### just RGB
    elif combination_name=='ExGI':
        return (np.where((ms_imaga_np[:,:,0]+ms_imaga_np[:,:,1]+ms_imaga_np[:,:,2])==0., 0, (2*ms_imaga_np[:,:,1])-(ms_imaga_np[:,:,0]+ms_imaga_np[:,:,2]))+1)*127.5 ### just RGB
    elif combination_name=='GCC':
        return (np.where((ms_imaga_np[:,:,0]+ms_imaga_np[:,:,1]+ms_imaga_np[:,:,2])==0., 0, ms_imaga_np[:,:,1]/(ms_imaga_np[:,:,0]+ms_imaga_np[:,:,1]+ms_imaga_np[:,:,2]))+1)*127.5 ### just RGB
        ### Indices that address Atmospheric (and other) Effects
    elif combination_name=='EVI':
        return (np.where((ms_imaga_np[:,:,4]+ms_imaga_np[:,:,0]+ms_imaga_np[:,:,2])==0., 0, 2.5*((ms_imaga_np[:,:,4]-ms_imaga_np[:,:,0])/(ms_imaga_np[:,:,4]+(6*ms_imaga_np[:,:,0])-(7.5*ms_imaga_np[:,:,2])+1)))+1)*127.5 # NOT VISUAL
    elif combination_name=='ARVI':
        return (np.where((ms_imaga_np[:,:,4]+ms_imaga_np[:,:,0]+ms_imaga_np[:,:,2])==0., 0, (ms_imaga_np[:,:,4]-(ms_imaga_np[:,:,0]*ms_imaga_np[:,:,2]))/(ms_imaga_np[:,:,4]+(ms_imaga_np[:,:,0]*ms_imaga_np[:,:,2])))+1)*127.5
    elif combination_name=='GARI':
        return (np.where((ms_imaga_np[:,:,4]+ms_imaga_np[:,:,1]+ms_imaga_np[:,:,0]+ms_imaga_np[:,:,2])==0., 0, (ms_imaga_np[:,:,4]-(ms_imaga_np[:,:,1]-(1.7*(ms_imaga_np[:,:,2]-ms_imaga_np[:,:,0]))))/(ms_imaga_np[:,:,4]+(ms_imaga_np[:,:,1]-(1.7*(ms_imaga_np[:,:,2]-ms_imaga_np[:,:,0])))))+1)*127.5
    elif combination_name=='VARI':
        return (np.where((ms_imaga_np[:,:,0]+ms_imaga_np[:,:,1]+ms_imaga_np[:,:,2])==0., 0, (ms_imaga_np[:,:,1]-ms_imaga_np[:,:,0])/(ms_imaga_np[:,:,1]+ms_imaga_np[:,:,0]-ms_imaga_np[:,:,2]))+1)*127.5### just RGB
        ### Addressing Soil Reflectance
        #SAVI requires a priori information about vegetation presence in the study area. We do not have that
        #MSAVI a modified SAVI (MSAVI) replaces 𝐿 factor in the SAVI equation (8) with a variable 𝐿 function. Based on the implementation of the MSAVI, Richardson
            #and Wiegand (1977) proposed a Modified Secondary Soil-
            #Adjusted Vegetation Index (MSAVI2)(Xue et al. 2017)
    elif combination_name=='MSAVI12':
        return (np.where((ms_imaga_np[:,:,4]+ms_imaga_np[:,:,0])==0., 0, (0.5*((2*ms_imaga_np[:,:,4]+1)-np.sqrt(np.power((2*ms_imaga_np[:,:,4]+1),2)-(8*(ms_imaga_np[:,:,4]-ms_imaga_np[:,:,0]))))))+1)*127.5
    elif combination_name=='MCARI':
        return (np.where((ms_imaga_np[:,:,0]+ms_imaga_np[:,:,3]+ms_imaga_np[:,:,1])==0., 0, ((ms_imaga_np[:,:,3]-ms_imaga_np[:,:,0])-0.2*(ms_imaga_np[:,:,3]-ms_imaga_np[:,:,1])*(ms_imaga_np[:,:,3]/ms_imaga_np[:,:,0])))+1)*127.5
    elif combination_name=='SIPI':
        return (np.where((ms_imaga_np[:,:,4]+ms_imaga_np[:,:,2]+ms_imaga_np[:,:,0])==0., 0, ((ms_imaga_np[:,:,4]-ms_imaga_np[:,:,2])/(ms_imaga_np[:,:,4]-ms_imaga_np[:,:,0])))+1)*127.5 # NOT VISUAL
        #Agricultural Indices
    elif combination_name=='TGI':
        return (np.where((ms_imaga_np[:,:,0]+ms_imaga_np[:,:,1]+ms_imaga_np[:,:,2])==0., 0, 0.5*(((650-450)*(ms_imaga_np[:,:,0]-ms_imaga_np[:,:,1]))-((650-560)*(ms_imaga_np[:,:,0]-ms_imaga_np[:,:,2]))))+1)*127.5### just RGB
    elif combination_name=='GLI':
        return (np.where((ms_imaga_np[:,:,0]+ms_imaga_np[:,:,1]+ms_imaga_np[:,:,2])==0., 0, (((ms_imaga_np[:,:,1]-ms_imaga_np[:,:,0])+(ms_imaga_np[:,:,1]-ms_imaga_np[:,:,2]))/((2*ms_imaga_np[:,:,1])+(ms_imaga_np[:,:,2]+ms_imaga_np[:,:,0]))))+1)*127.5### just RGB
        #Task-specific Vegetation Indices
        #(TDVI) was developed to detect vegetation in urban settings where NDVI is often saturated. 
    elif combination_name=='TDVI':
        return (np.where((ms_imaga_np[:,:,4]+ms_imaga_np[:,:,0])==0., 0, 1.5*(ms_imaga_np[:,:,4]-ms_imaga_np[:,:,0])/np.sqrt(np.power(ms_imaga_np[:,:,4],2)+ms_imaga_np[:,:,0]+0.5))+1)*127.5
    elif combination_name=='NDVI':
        return (np.where((ms_imaga_np[:,:,4]+ms_imaga_np[:,:,0])==0., 0, np.divide((ms_imaga_np[:,:,4]-ms_imaga_np[:,:,0]),(ms_imaga_np[:,:,4]+ms_imaga_np[:,:,0])))+1)*127.5
    elif combination_name=='GNDVI':
        return (np.where((ms_imaga_np[:,:,4]+ms_imaga_np[:,:,1])==0., 0, (ms_imaga_np[:,:,4]-ms_imaga_np[:,:,1])/(ms_imaga_np[:,:,4]+ms_imaga_np[:,:,1]))+1)*127.5
    elif combination_name=='NDRE':
        return (np.where((ms_imaga_np[:,:,4]+ms_imaga_np[:,:,3])==0., 0, (ms_imaga_np[:,:,4]-ms_imaga_np[:,:,3])/(ms_imaga_np[:,:,4]+ms_imaga_np[:,:,3]))+1)*127.5
    elif combination_name=='LCI':
        return (np.where((ms_imaga_np[:,:,4]+ms_imaga_np[:,:,3]+ms_imaga_np[:,:,0])==0., 0, (ms_imaga_np[:,:,4]-ms_imaga_np[:,:,3])/(ms_imaga_np[:,:,4]+ms_imaga_np[:,:,0]))+1)*127.5
    else:
        raise Exception("Vegetation index provided not found")
def is_vegetation_index(bandName):
    indexes = ['RVI','DVI','VDVI','ExGI','GCC','ARVI','GARI','VARI','MSAVI12','MCARI','TGI','GLI','TDVI','NDVI','GNDVI','NDRE','LCI']
    return bandName in indexes