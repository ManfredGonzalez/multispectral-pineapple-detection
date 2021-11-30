import cv2
import numpy as np
import os
import rasterio

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

'''
This method is useful to split or slice an image into different pieces
sliceWidth, sliceHeight:
    These are the params that set the height and width of the slices in which the
    original image will be divided.
'''
def myImageSlicer(sliceWidth,sliceHeight,img,results_path=None, ortomosaic_name=None, band_name=None,file_type='.JPG'):
    exc = [(4,13),(6,12),(7,4),(8,11),(11,8),(14,0),(15,1),(15,2)]
    y = 0
    x = 0
    width = sliceWidth
    height = sliceHeight
    ai1 = 0
    ai2 = 0
    originalHeight = img.shape[0]
    originalWidth = img.shape[1]
    images_matrix = np.empty(shape=(int(originalHeight/sliceHeight),int(originalWidth/sliceWidth))+(0,)).tolist()
    while y < originalWidth:
        while x < originalHeight:
            crop_img = img[x:x+height, y:y+width]
            images_matrix[ai2][ai1] = crop_img
            if results_path:
                if np.sum(crop_img) != 0 and np.sum(crop_img) != 0.0 and (ai2,ai1) not in exc:
                    crop_img = normalize(crop_img, 0, 127).astype('uint8')
                    if band_name:
                        image_path = os.path.join(results_path,f'{ortomosaic_name}_{str(ai2)}_{str(ai1)}_{band_name}{file_type}')
                    else:
                        image_path = os.path.join(results_path,f'{ortomosaic_name}_{str(ai2)}_{str(ai1)}{file_type}')
                    print(f'Image saved at: {image_path}')
                    cv2.imwrite(image_path, crop_img)
            x = x + height
            ai2 = ai2 + 1
        
        x = 0
        y = y + width
        ai2 = 0
        ai1 = ai1 + 1

    return images_matrix


results_path = '/mnt/e/datosmanfred/mosaicos/L301_B21_y_3_20m/L301_B21_y_3_20m_cropped'
ortomosaic_name = 'L301_B21_y_3_20m'
band_name = 'NIR'
if band_name:
    mosaic_path = f'/mnt/e/datosmanfred/mosaicos/L301_B21_y_3_20m/{ortomosaic_name}_{band_name}.tif'
else:
    mosaic_path = f'/mnt/e/datosmanfred/mosaicos/L301_B21_y_3_20m/{ortomosaic_name}.tif'
if not os.path.exists(results_path):
    os.mkdir(results_path)

#array = cv2.imread(mosaic_path,cv2.COLOR_BGR2RGB)
array = rasterio.open(mosaic_path).read(1)
array = np.nan_to_num(array)
base_name,file_type = os.path.splitext(mosaic_path)

myImageSlicer(621,792,array,results_path=results_path, ortomosaic_name=ortomosaic_name,band_name=band_name,file_type='.TIF')
print(array.shape)
print(base_name)
print(ortomosaic_name)