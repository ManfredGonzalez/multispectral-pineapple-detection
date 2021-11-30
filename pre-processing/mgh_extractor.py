import cv2
import numpy as np
import os
from numba import jit

'''
This method is useful to split or slice an image into different pieces
sliceWidth, sliceHeight:
    These are the params that set the height and width of the slices in which the
    original image will be divided.
'''
def myImageSlicer(sliceWidth,sliceHeight,img):
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
            #cv2.imwrite('slice_0'+str(ai2)+'_0'+str(ai1)+'.jpg', crop_img)
            x = x + height
            ai2 = ai2 + 1
        
        x = 0
        y = y + width
        ai2 = 0
        ai1 = ai1 + 1

    return images_matrix

class Mgh_feature_extractor():
    def __init__(self,image):
        self.sift = cv2.SIFT_create()
        self.image = image
        '''if self.image.shape != self.image_2.shape:
            raise Exception(f'Images shapes do not match, image1: {self.image.shape} != image2: {self.image_2.shape}')'''
        self.width = self.image.shape[1]
        self.height = self.image.shape[0]

        self.best_slice_width = 0
        self.best_slice_height = 0
        ### There would be a bug if the width or height of the image are prime numbers since the best width or values will be the same as the original dimensions
        for i in range(4,self.width+1):
            if self.width % i == 0:
                self.best_slice_width = int(self.width/i)
                break
        for i in range(4,self.height+1):
            if self.height % i == 0:
                self.best_slice_height = int(self.height/i)
                break

    def getFeatures(self):
        im_sliced_1 = myImageSlicer(self.best_slice_width, self.best_slice_height,self.image)
        #keypoints_list = []
        keypoints_list = ()
        descriptors_list = []
        for i in range(int(self.width/self.best_slice_width)):
            for j in range(int(self.height/self.best_slice_height)):
                keypoints, descriptors = self.sift.detectAndCompute(im_sliced_1[i][j], None)
                for k in range(len(keypoints)):
                    point_with_offset = (keypoints[k].pt[0]+((j)*int(self.best_slice_width)),keypoints[k].pt[1]+((i)*int(self.best_slice_height)))
                    keypoints[k].pt = point_with_offset
                #keypoints_list.append(keypoints)
                keypoints_list = keypoints_list + keypoints
                descriptors_list.append(descriptors)
        descriptors_concatenated = np.concatenate(descriptors_list)
        return keypoints_list, descriptors_concatenated

'''extractor_im1 = Mgh_feature_extractor('/mnt/e/datosmanfred/Gira 10 13 Mar21/Lote 71/5_meters/00000001.JPG')
extractor_im2 = Mgh_feature_extractor('/mnt/e/datosmanfred/Gira 10 13 Mar21/Lote 71/5_meters/00000001_Green.TIF')
draw_matches = False
keypoints_1, descriptors1 = extractor_im1.getFeatures()
keypoints_2, descriptors2 = extractor_im2.getFeatures()



# Match features.
#bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
bf = cv2.BFMatcher(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING, crossCheck=True)

matches = bf.match(descriptors1,descriptors2)
matches = sorted(matches, key = lambda x:x.distance)
if draw_matches:
    imMatches = cv2.drawMatches(extractor_im1.image, keypoints_1, extractor_im2.image, keypoints_2, matches, None)
    cv2.imwrite(f"matchesCrossed.jpg", imMatches)

print(len(matches))
print('Done')'''
'''
draw_matches = False
im1Gray = cv2.imread('/mnt/e/datosmanfred/Gira 10 13 Mar21/Lote 71/5_meters/00000001.JPG', cv2.IMREAD_GRAYSCALE)
im2Gray = cv2.imread('/mnt/e/datosmanfred/Gira 10 13 Mar21/Lote 71/5_meters/00000001_Green.TIF', cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT_create()
# Detect ORB features and compute descriptors.

keypoints1, descriptors1 = sift.detectAndCompute(im1Gray, None)
keypoints2, descriptors2 = sift.detectAndCompute(im2Gray, None)


# Match features.
#bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
bf = cv2.BFMatcher(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING, crossCheck=True)
#bf = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

matches = bf.match(descriptors1,descriptors2,None)
matches = sorted(matches, key = lambda x:x.distance)
print(len(matches))
# Remove not so good matches
#numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
#matches = matches[:numGoodMatches]

# Draw top matches
if draw_matches:
    imMatches = cv2.drawMatches(im1Gray, keypoints1, im2Gray, keypoints2, matches, None)
    cv2.imwrite("matchesCOMPLETE.jpg", imMatches)'''
        
        