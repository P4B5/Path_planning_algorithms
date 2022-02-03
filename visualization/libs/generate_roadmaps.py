import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os

# =========================================================
# FUNCTION TO CONVERT GREYSCALE MAPS INTO BINARY MAPS OF 0s AND 1s. 
# Parameters: 
#   - img_path:str, absolute path to the input_image. Initial None.
#   - input_image:np.ndarray, the preloaded input image. Initial None.
#   - thresh:int, binary threshold. All image_intensities > thresh --> 1. Initial 250.
#   - morphology_kernel:np.ndarray, kernel used in morphology transforms. Initial np.ones((5,5), np.uint8).
# Returns:
#   - img_bin:np.ndarray, output image, consisting of 0s and 1s. 
def binary_threshold_image(image_path:str = None, input_img:np.ndarray = None, thresh:int = 245, morphology_kernel:np.ndarray = np.ones((1,1), np.uint8)):
    # Assert that input is given
    if input_img is None: 
        assert image_path != None, "Input not defined for thresholding"
        # Read image you want to converge to a roadmap
        img = cv.imread(image_path)
    if image_path is None: 
        assert input_img.all() != None, "Input not defined for thresholding"
        # Use input image
        img = input_img

    # Convert to greyscale and equalize histogram for greater contrast
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray = cv.equalizeHist(img_gray)
    # Threshold the image into binary intensities given _thresh_
    _, img_bin = cv.threshold(img_gray, thresh, 255, cv.THRESH_BINARY)

    # Dilate and erode image to remove holes, unwanted lines ect.
    img_bin = cv.dilate(img_bin, morphology_kernel, iterations = 1)
    img_bin = cv.erode(img_bin, morphology_kernel, iterations = 1)
    
    # Change to binary, setting 255's to 1's
    #img_bin[img_bin == 255] = 1

    return img_bin

# =========================================================
# FUNCTION TO SHOW MAPS AS IMAGES
# Parameters: 
#   - img_bin:np.ndarray, Binary map as a 2D matrix. initial None
#   - img_orig:np.ndarray, The original map, as a 2D matrix. initial None
# Returns:
#   None
def print_image(img_bin = None, img_orig = None):
    if img_orig is None and img_bin is None:
        print("PRINTING ERROR: Select image for printing")
        return
    # Generate a 1x2 subplot for both images
    if img_orig is not None and img_bin is not None: 
        _, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(img_orig, cmap='gray')
        ax1.set_title('Original map')
        ax1.axis('off')
        ax2.imshow(img_bin, cmap='gray')
        ax2.set_title('Binary map')
        ax2.axis('off')
        plt.show()
        return
    # Show only the binary image
    if img_orig is None and img_bin is not None: 
        plt.imshow(img_bin, cmap='gray')
        plt.title('Binary map')
        plt.axis('off')
        plt.show()
        return
    # Show only the original image
    if img_orig is not None and img_bin is None: 
        plt.imshow(img_orig, cmap='gray')
        plt.title('Desired map')
        plt.axis('off')
        plt.show()
        return