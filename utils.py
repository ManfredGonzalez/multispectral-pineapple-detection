import bs4
from bs4 import BeautifulSoup
import cv2
import numpy as np
def getXMPData(filePath):
    fin = open( filePath, "rb")
    img = fin.read()
    imgAsString=str(img)
    xmp_start = imgAsString.find('<x:xmpmeta')
    xmp_end = imgAsString.find('</x:xmpmeta')
    if xmp_start != xmp_end:
        xmpString = imgAsString[xmp_start:xmp_end+12]
    
    xmpAsXML = BeautifulSoup(xmpString, "xml")
    info = str(xmpAsXML.Description).split('drone-dji:')
    info_dji = info[1:len(info)-1]
    keys = []
    values = []
    for line in info_dji:
        line = line.replace('"',"")
        line = line.replace('\\n',"")
        line = line.replace(' ',"")
        line = line.split('=')
        keys.append(line[0])
        values.append(line[1])
    dictionary = dict(zip(keys,values))
    return dictionary

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def myImageSlicer(sliceWidth,sliceHeight, originalWidth, originalHeight,img):
    y = 0
    x = 0
    width = sliceWidth
    height = sliceHeight
    ai1 = 0
    ai2 = 0
    #img = cv2.imread(imagePath)
    #originalHeight = img.shape[0]
    #originalWidth = img.shape[1]
    images_matrix = np.empty(shape=(int(originalHeight/sliceHeight),int(originalWidth/sliceWidth))+(0,)).tolist()
    while y < originalWidth:
        while x < originalHeight:
            crop_img = img[x:x+height, y:y+width]
            images_matrix[ai2][ai1] = crop_img
            #cv2.imwrite('slice_0'+str(ai2)+'_0'+str(ai1)+'.jpg', crop_img)
            x = x + height
            ai2 = ai2 + 1
        
        x = 0
        y = y + width
        ai2 = 0
        ai1 = ai1 + 1

    return images_matrix

'''import rasterio
image_path = 'multispectral-pineapple-detection/datasets/mosaico/L301 B24_10m/L301_B24_10m.tif'
image = rasterio.open(image_path)
red = image.read(1).astype('uint8')
green = image.read(2).astype('uint8')
blue = image.read(3).astype('uint8')
rgb = np.dstack((red,green,blue))
slicedMosaic = myImageSlicer(660,865,6600,8650,rgb)
print(type(slicedMosaic))'''