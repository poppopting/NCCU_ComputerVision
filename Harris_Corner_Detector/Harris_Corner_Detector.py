#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt

class Harris_Corner_Detector():
    def __init__(self, deriv_way='1stCentral', window_size=3, thresold=1000):
        self.deriv_way = deriv_way
        self.window_size = window_size
        self.thresold = thresold
        
    def __grad_compute(self, img):
        img = np.float32(img)
        self.shape = img.shape
        dx = np.zeros(self.shape)
        dy = np.zeros(self.shape)
        
        # use forward or backward to copmute grdient of pixel in boudaries 
        dx[:,0] = img[:,1] - img[:,0]
        dx[:,-1] = img[:,-2] - img[:,-1]
        dy[0,:] = img[1,:] - img[0,:]
        dy[-1,:] = img[-2,:] - img[-1,:]
        
        if self.deriv_way == '1stCentral':
            dx[:,1:-1] = (img[:,2:] - img[:,:-2]) / 2
            dy[1:-1,:] = (img[2:,:] - img[:-2,:]) / 2
            
        elif self.deriv_way == '1stForward':
            dx[:,1:-1] = img[:,2:] - img[:,1:-1] 
            dy[1:-1,:] = img[2:,:] - img[1:-1,:] 
        
        elif self.deriv_way == '1stBackward':
            dx[:,1:-1] = img[:,1:-1] - img[:,:-2]
            dy[1:-1,:] = img[1:-1,:] - img[:-2,:]
            
        else:
            return print('There are currently only 3 ways to calculate differential: "1stCenter", "1stForward", "1stBackward".')
            
        return dx, dy
    
    def apply(self, img):
        self.img = img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dx, dy = self.__grad_compute(gray)
        Ixx = dx**2
        Iyy = dy**2
        Ixy = dx*dy
        
        f = np.zeros(self.shape)
        y_sha, x_sha = self.shape
        for y in range(y_sha):
             for x in range(x_sha):
                    if (y < y_sha-self.window_size) & (x < x_sha-self.window_size):
                        Hxx = Ixx[y:y+self.window_size, x:x+self.window_size].sum(axis=(0,1))
                        Hyy = Iyy[y:y+self.window_size, x:x+self.window_size].sum(axis=(0,1))
                        Hxy = Ixy[y:y+self.window_size, x:x+self.window_size].sum(axis=(0,1))

                    else:
                        Hxx = Ixx[y-self.window_size+1:y+1, x-self.window_size+1:x+1].sum(axis=(0,1))
                        Hyy = Iyy[y-self.window_size+1:y+1, x-self.window_size+1:x+1].sum(axis=(0,1))
                        Hxy = Ixy[y-self.window_size+1:y+1, x-self.window_size+1:x+1].sum(axis=(0,1))

                    det = Hxx*Hyy - Hxy**2
                    tr = Hxx + Hyy
                    f[y,x] = det / tr if tr != 0 else 0
            
        self.red_img = img.copy()
        self.red_img[f>self.thresold,:] = np.array([0,0,255])
        
        self.f_img = gray.copy()     
        self.f_img[f<=self.thresold] = 0
        self.f_img[f>self.thresold] = 255
        
        return self #self.red_img, self.f_img
    
    def plot_features(self):
        
        plt.figure(figsize=(20,18))
        plt.subplot(1,3,1)
        plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.subplot(1,3,2)
        plt.imshow(self.f_img, cmap='gray')
        plt.title("Different ways: {0}, Window_size: {1}, Threshold: {2}".format(self.deriv_way, self.window_size, self.thresold))
        plt.subplot(1,3,3)
        plt.imshow(cv2.cvtColor(self.red_img, cv2.COLOR_BGR2RGB))
        plt.title('Features in Image')
        plt.show()
        
        return self


# read image
giraffe = cv2.imread('giraffe.jpg')
tailwind = cv2.imread('Tailwind79.jpg')

# Construct Harris_Corner_Detector
harris_cent_5_1500 = Harris_Corner_Detector(deriv_way='1stCentral', window_size=5, thresold=1500)
harris_cent_5_1800 = Harris_Corner_Detector(deriv_way='1stCentral', window_size=5, thresold=1800)
harris_cent_7_1500 = Harris_Corner_Detector(deriv_way='1stCentral', window_size=7, thresold=1500)
harris_cent_7_1800 = Harris_Corner_Detector(deriv_way='1stCentral', window_size=7, thresold=1800)
harris_back_5_1500 = Harris_Corner_Detector(deriv_way='1stBackward', window_size=5, thresold=1500)
harris_back_5_1800 = Harris_Corner_Detector(deriv_way='1stBackward', window_size=5, thresold=1800)
harris_back_7_1500 = Harris_Corner_Detector(deriv_way='1stBackward', window_size=7, thresold=1500)
harris_back_7_1800 = Harris_Corner_Detector(deriv_way='1stBackward', window_size=7, thresold=1800)

# apply Harris_Corner_Detector on giraffe
harris_cent_5_1500.apply(giraffe).plot_features()
harris_cent_5_1800.apply(giraffe).plot_features()
harris_cent_7_1500.apply(giraffe).plot_features()
harris_cent_7_1800.apply(giraffe).plot_features()
harris_back_5_1500.apply(giraffe).plot_features()
harris_back_5_1800.apply(giraffe).plot_features()
harris_back_7_1500.apply(giraffe).plot_features()
harris_back_7_1800.apply(giraffe).plot_features()

# apply Harris_Corner_Detector on tailwind
harris_cent_5_1500.apply(tailwind).plot_features()
harris_cent_5_1800.apply(tailwind).plot_features()
harris_cent_7_1500.apply(tailwind).plot_features()
harris_cent_7_1800.apply(tailwind).plot_features()
harris_back_5_1500.apply(tailwind).plot_features()
harris_back_5_1800.apply(tailwind).plot_features()
harris_back_7_1500.apply(tailwind).plot_features()
harris_back_7_1800.apply(tailwind).plot_features()

