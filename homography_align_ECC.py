import cv2
import numpy as np
import rasterio
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

def alignImages(im1, im2):
    # Find size of image1
    sz = im1.shape

    im1_gray = im1
    im2_gray = im2
    # Define the motion model
    #warp_mode = cv2.MOTION_TRANSLATION
    warp_mode = cv2.MOTION_HOMOGRAPHY

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 5000

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(im1_gray,im2_gray,warp_matrix, warp_mode, criteria)

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
    # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective (im1, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
    # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im1, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    # Show final results
    return im2_aligned, warp_matrix

if __name__ == '__main__':    
    # Read reference image
    refFilename = "data/multispectral_image/00000001.jpg"
    print("Reading reference image : ", refFilename)
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
    # Shift the channels to RGB
    imReference = cv2.cvtColor(imReference, cv2.COLOR_BGR2RGB)
    imReference = cv2.cvtColor(imReference, cv2.COLOR_RGB2GRAY)

    # Read the different bands
    imFilename = "data/multispectral_image/00000001_Green.TIF"
    green = normalize(rasterio.open(imFilename).read(1),0,255).astype('uint8')
    imFilename = "data/multispectral_image/00000001_Blue.TIF"
    blue = normalize(rasterio.open(imFilename).read(1),0,255).astype('uint8')
    imFilename = "data/multispectral_image/00000001_Red.TIF"
    red = normalize(rasterio.open(imFilename).read(1),0,255).astype('uint8')

    imFilename = "data/multispectral_image/00000001_RE.TIF"
    #re = normalize(rasterio.open(imFilename).read(1),0,255).astype('uint8')
    imFilename = "data/multispectral_image/00000001_NIR.TIF"
    #nir = normalize(rasterio.open(imFilename).read(1),0,255).astype('uint8')

    print("Aligning images ...")
    # Registered image will be resotred in imReg.
    # The estimated homography will be stored in h.
    #imReg, h = alignImages(im, imReference)
    greenReg, h_green = alignImages(green, imReference)
    blueReg, h_blue = alignImages(blue, imReference)
    redReg, h_red = alignImages(red, imReference)
    #reReg, h_re = alignImages(re, imReference)
    #nirReg, h_re = alignImages(nir, imReference)

    # Write aligned image to disk.
    outFilename = "aligned.jpg"
    print("Saving aligned image : ", outFilename)
    cv2.imwrite('greenReg.jpg', greenReg)
    cv2.imwrite('blueReg.jpg', blueReg)
    cv2.imwrite('redReg.jpg', redReg)
    #cv2.imwrite('reReg.jpg', reReg)
    #cv2.imwrite('nirReg.jpg', nirReg)

    im = np.dstack((redReg, greenReg, blueReg))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    #im = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
    cv2.imwrite('RGB.jpg', im)
    # Print estimated homography
    #print("Estimated homography : \n",  h)