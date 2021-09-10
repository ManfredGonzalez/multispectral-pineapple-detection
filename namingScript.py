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
    fin = open( filePath, "rb")
    img = fin.read()
    imgAsString=str(img)
    xmp_start = imgAsString.find('<x:xmpmeta')
    xmp_end = imgAsString.find('</x:xmpmeta')
    if xmp_start != xmp_end:
        xmpString = imgAsString[xmp_start:xmp_end+12]
    
    xmpAsXML = BeautifulSoup(xmpString, "xml")
    #print(type(xmpAsXML.prettify()))
    #print(xmpAsXML.prettify())
    root = ET.fromstring(xmpAsXML.prettify())
    #print(xmpAsXML.prettify())
    bandName = root.findall('.//BandName')[0].text.strip()
    return bandName



dir_of_images_path = 'D:/Manfred/InvestigacionPinas/Beca-CENAT/workspace/multispectral-hiperspectral/Gira 10 13 Mar21/Lote 71/132MEDIA'
myImages = glob.glob(dir_of_images_path+'/*.JPG')
imagesNamesRange = range(68,100000000)  ####### Specify the starting number 
imagecounter = 0;
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

