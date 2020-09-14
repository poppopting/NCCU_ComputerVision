#!/usr/bin/env python
# coding: utf-8


import numpy as np
import cv2


def HistogramEqualization_2d(img, L):

    img_flatten = img.flatten()
    hist, _ = np.histogram(img_flatten, bins=np.arange(L+1))
    pdf = hist / sum(hist)
    cdf = pdf.cumsum()
    transform = np.ceil((L-1) * cdf)
    proceed = transform[img_flatten].reshape(img.shape).astype('uint8')
    
    return proceed


### Q1####

# read the image
mp2 = cv2.imread('mp2.jpg', cv2.IMREAD_GRAYSCALE)

Q1 = HistogramEqualization_2d(mp2, L=256)
# Q1
cv2.imwrite('HE_mp2.jpg', Q1)
# Q1 opencv
Q1_cv = cv2.equalizeHist(mp2)
cv2.imwrite('HE_mp2_cv.jpg', Q1_cv)


### Q2 ###

# convert color 
mp2a_BGR = cv2.imread('mp2a.jpg', cv2.IMREAD_COLOR)
mp2a_hsv = cv2.cvtColor(mp2a_BGR, cv2.COLOR_BGR2HSV)
mp2a_YCbCr = cv2.cvtColor(mp2a_BGR, cv2.COLOR_BGR2YCR_CB)

############## BGR #########################################################
mp2a_BGR_Q2 = mp2a_BGR.copy()
for color in range(3):
    mp2a_BGR_Q2[:,:,color] = HistogramEqualization_2d(mp2a_BGR_Q2[:,:,color], L=256)
cv2.imwrite('HE_mp2a_RGB.jpg', mp2a_BGR_Q2)
# (a) BGR opencv
mp2a_BGR_cv = mp2a_BGR.copy()
for color in range(3):
    mp2a_BGR_cv[:,:,color] = cv2.equalizeHist(mp2a_BGR_cv[:,:,color])
cv2.imwrite('HE_mp2a_RGB_cv.jpg', mp2a_BGR_cv)


############## HSV #########################################################
mp2a_hsv_Q2 = mp2a_hsv.copy()
mp2a_hsv_Q2[:,:,2] = HistogramEqualization_2d(mp2a_hsv_Q2[:,:,2], L=256)
# convert channel back to BGR
mp2a_hsv_Q2 = cv2.cvtColor(mp2a_hsv_Q2, cv2.COLOR_HSV2BGR) 
cv2.imwrite('HE_mp2a_HSV.jpg', mp2a_hsv_Q2)
# (b) HSV opencv
mp2a_hsv_cv = mp2a_hsv.copy()
mp2a_hsv_cv[:,:,2] = cv2.equalizeHist(mp2a_hsv_cv[:,:,2])
# convert channel back to BGR
mp2a_hsv_cv = cv2.cvtColor(mp2a_hsv_cv, cv2.COLOR_HSV2BGR) 
cv2.imwrite('HE_mp2a_HSV_cv.jpg', mp2a_hsv_cv)


############## YCbCr #########################################################
mp2a_YCbCr_Q2 = mp2a_YCbCr.copy()
mp2a_YCbCr_Q2[:,:,0] = HistogramEqualization_2d(mp2a_YCbCr_Q2[:,:,0], L=256)
# convert channel back to BGR
mp2a_YCbCr_Q2 = cv2.cvtColor(mp2a_YCbCr_Q2, cv2.COLOR_YCR_CB2BGR)  
cv2.imwrite('HE_mp2a_YCbCr.jpg', mp2a_YCbCr_Q2)
# (c) YCbCr opencv
mp2a_YCbCr_cv = mp2a_YCbCr.copy()
mp2a_YCbCr_cv[:,:,0] = cv2.equalizeHist(mp2a_YCbCr_cv[:,:,0])
# convert channel back to BGR
mp2a_YCbCr_cv = cv2.cvtColor(mp2a_YCbCr_cv, cv2.COLOR_YCR_CB2BGR)  
cv2.imwrite('HE_mp2a_YCbCr_cv.jpg', mp2a_YCbCr_cv)





