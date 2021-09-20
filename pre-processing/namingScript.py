# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 17:50:26 2021

@author: Manfred
"""

import bs4
from bs4 import BeautifulSoup
import xml
from xml.etree import ElementTree as ET
import glob
import os
from utils import getXMPData

def numberName(number,totalDigits):
    additionalCeros = totalDigits-len(str(number))
    if additionalCeros < 0:
        return "number over the specified range"
    name = ""
    for digit in range(additionalCeros):
        name = "0" + name
    name = name + str(number)
    return name

def getBandName(filePath):
    meta_data = getXMPData(filePath)
    return meta_data['BandName']



dir_of_images_path = 'gira_30Jun_al_2Jul_2021_10m/L301 B24/103FPLAN'
myImages = glob.glob(dir_of_images_path+'/*.JPG')
imagesNamesRange = range(259,100000000)  ####### Specify the starting number

imagecounter = 0
for rgb_image in myImages:
    imageName = rgb_image[len(dir_of_images_path)+1:len(rgb_image)-4]
    imageNumber = int(imageName[4:8])
    imageNewName = numberName(imagesNamesRange[imagecounter],8)
    imagecounter = imagecounter + 1
    os.rename(f'{rgb_image}',f'{dir_of_images_path}/{imageNewName}.JPG')
    for tiffImageNumber in range(imageNumber+1,imageNumber+6):
        tiff_oldFileOldPath = dir_of_images_path+'/'+'DJI_'+numberName(tiffImageNumber,4)+'.TIF'
        
        bandName = getBandName(tiff_oldFileOldPath)
        if bandName == 'RedEdge':
            bandName = 'RE'
        tiff_imageNewName = f'{imageNewName}_{bandName}.TIF'
        #print(f'{imageNewName}_{bandName}')
        os.rename(f'{tiff_oldFileOldPath}',f'{dir_of_images_path}/{tiff_imageNewName}')

