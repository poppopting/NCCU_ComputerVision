#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt


# Implement convolution by OpenCV 
def OPENCV_FILTER(image, kernel):
    dst = cv2.filter2D(image, -1, kernel)
    
    return dst 


# Implement convolution by Kuan-Ting Chen 
def MY_FILTER(image, kernel):
    x, y = image.shape
    ker_0, ker_1 = kernel.shape
    # this vesrion only valid when each dimemsion of kernal size be odd
    if (ker_0 %2 == 0) or (ker_1 %2 == 0):
        print('Each dimemsion of kernal size should be odd !!!')
        print("Image won't be modified.")
        return image
    
    # use reflection to pad the array
    # this would ensure the output image size equal to input image size 
    pad_0, pad_1 = ((np.array([ker_0, ker_1]) - 1) / 2).astype(int)
    pad_img = np.pad(image, ((pad_0, pad_0), (pad_1, pad_1)), 'reflect')
    
    # begin convoluting
    filtered = np.zeros(shape=(x, y))
    for i in range(x):
        for j in range(y):
            # sum along first and second dimmesion 
            filtered[i, j] = np.sum(pad_img[i:i+ker_0, j:j+ker_1] * kernel)
    
    # convert the values of each dimension to 0~255  
    # first we scale values to 0~1 and then mutiply by 255
    filter_minima = np.min(filtered)
    filter_maxima = np.max(filtered)
    filtered = 255 * (filtered - filter_minima) / (filter_maxima - filter_minima)
    
    # convert dtype to uint8 to ensure the values are integers between 0 and 255     
    filtered = filtered.astype('uint8')

    return filtered 



# define function to show all of our output images
def show_pic(pic_dict):
    plt.figure(figsize=(12, 6))
    len_ =  len(pic_dict)
    for i, (title, img) in enumerate(pic_dict.items()): 
        plt.subplot(1, len_, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
    plt.show()



##############################################################

# read the image
original_img = cv2.imread('Lenna.jpg', cv2.IMREAD_GRAYSCALE)


# the result of convolution implement by me and opencv
# their effects are similar but color are a little different
kernel = np.ones((5,5)) / 25
filtered_cv = OPENCV_FILTER(original_img, kernel)
filtered_my = MY_FILTER(original_img, kernel)
show_pic({'Original Image': original_img,
          'filtered by OpenCV': filtered_cv,
          'filtered by my function': filtered_my})


# the result of convolution implement by me and opencv
# their effects are similar but colors are a bit different
kernel = np.array([[0, -1, 0], 
                   [-1, 5, -1],
                   [0, -1, 0]])
filtered_cv = OPENCV_FILTER(original_img, kernel)
filtered_my = MY_FILTER(original_img, kernel)
show_pic({'Original Image': original_img,
          'filtered by OpenCV': filtered_cv,
          'filtered by my function': filtered_my})

