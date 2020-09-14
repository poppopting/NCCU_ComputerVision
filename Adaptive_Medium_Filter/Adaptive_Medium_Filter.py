#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt


### Q1

bridge = cv2.imread('bridge.jpg', cv2.IMREAD_GRAYSCALE)


# (a)
def get_quarter_largest(img):
    x, y = img.shape
    
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    mag = cv2.magnitude(dft[:,:,0], dft[:,:,1])
    threshold = np.quantile(mag.flatten(), 0.75)
    dft[mag <= threshold, :] = 0
    dft_inv = cv2.idft(dft, flags=cv2.DFT_SCALE)
    mag_inv = cv2.magnitude(dft_inv[:,:,0], dft_inv[:,:,1])
    
    return mag_inv

A = get_quarter_largest(bridge)

plt.figure(figsize=(12,16))
plt.subplot(1,2,1)
plt.title('Original Image')
plt.imshow(bridge, cmap='gray')
plt.subplot(1,2,2)
plt.title('Q1 (a) Image')
plt.imshow(A, cmap='gray')
plt.show()


# (b)

x, y = bridge.shape
B = np.zeros((x,y))
for i in range(0, x, 16):
    for j in range(0, y, 16):
        subimg = bridge[i:i+16, j:j+16].copy()
        B[i:i+16, j:j+16] = get_quarter_largest(subimg)

plt.figure(figsize=(12,16))
plt.subplot(1,2,1)
plt.title('Original Image')
plt.imshow(bridge, cmap='gray')
plt.subplot(1,2,2)
plt.title('Q1 (b) Image')
plt.imshow(B, cmap='gray')
plt.show()


# (c)
# using averaging to reduce the size of the original image
def avarage_image(img):
    
    reduced_img = np.zeros((128,128))
    for i in range(0, 256, 2):
        for j in range(0, 256, 2):
            reduced_img[int(i/2), int(j/2)] = np.mean(img[i:i+2, j:j+2])
        
    return reduced_img

reduced_img = avarage_image(bridge)
dft_img = cv2.dft(reduced_img, flags = cv2.DFT_COMPLEX_OUTPUT)
dft_img_shift = np.fft.fftshift(dft_img)
pad_img = np.pad(dft_img_shift, ((64,64),(64,64),(0,0)), 'constant', constant_values=0)
pad_img_pack = np.fft.ifftshift(pad_img)
pad_inv = cv2.idft(pad_img_pack, flags=cv2.DFT_SCALE)
C = cv2.magnitude(pad_inv[:,:,0], pad_inv[:,:,1])

plt.figure(figsize=(12,16))
plt.subplot(1,2,1)
plt.title('Original Image')
plt.imshow(bridge, cmap='gray')
plt.subplot(1,2,2)
plt.title('Q1 (c) Image')
plt.imshow(C, cmap='gray')
plt.show()

## compare MSE
def MSE(img1, img2):
    
    return np.mean((img1 - img2)**2)

print('MSE between Image and Image A : {:.4f}'.format(MSE(bridge, A)))
print('MSE between Image and Image B : {:.4f}'.format(MSE(bridge, B)))
print('MSE between Image and Image C : {:.4f}'.format(MSE(bridge, C)))


### Q2

class Adaptive_Median_Filter():
    
    def __init__(self, ks_init=2, S_max=7):
        self.S_max = S_max
        self.ks_init = ks_init
        
    def __adaptive_median_filter(self, i, j, img):
        # current kernel size
        cur_ks = self.ks_init
        z_xy = img[i,j]
        # 以左上角當作定位點
        while cur_ks <= self.S_max:
            S_xy = img[i:i+cur_ks, j:j+cur_ks]
            z_min = np.min(S_xy)
            z_max = np.max(S_xy)
            z_med = np.median(S_xy)
            
            A1 = int(z_med) - int(z_min)
            A2 = int(z_med) - int(z_max)

            if (A1 > 0) and (A2 < 0):
                B1 = int(z_xy) - int(z_min)
                B2 = int(z_xy) - int(z_max)

                if (B1 > 0) and (B2 < 0):
        
                    return z_xy
                else:
                   
                    return z_med
            else:
                cur_ks += 1
          
        return z_xy
        
    def apply(self, img):
        
        x, y = img.shape
        proceed_img = np.zeros((x,y))
        
        for i in range(x):
            for j in range(y):
                proceed_img[i,j] = self.__adaptive_median_filter(i, j, img)
                
        return proceed_img.astype('uint8')            

fig_0514a = cv2.imread('Fig0514(a)(ckt_saltpep_prob_pt25).tif', cv2.IMREAD_GRAYSCALE)

AMF = Adaptive_Median_Filter(ks_init=3, S_max=7)
AMF_img = AMF.apply(fig_0514a)

plt.figure(figsize=(12,16))
plt.subplot(1,2,1)
plt.title('Original Image')
plt.imshow(fig_0514a, cmap='gray')
plt.subplot(1,2,2)
plt.title('Q2 Image')
plt.imshow(AMF_img, cmap='gray')
plt.show()


### Q3 (b)

def median_filter_on_freq(img, ks=2):
    x, y = img.shape
    dft_img = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft_img)
    proceed_img = np.zeros((x,y,2))
    
    for i in range(0,x,ks):
        for j in range(0,y,ks):
            proceed_img[i,j,0] = np.median(dft_shift[i:i+ks, j:j+ks,0])
            proceed_img[i,j,1] = np.median(dft_shift[i:i+ks, j:j+ks,1])
            
    img_ishift = np.fft.fftshift(proceed_img)
    img_idft = cv2.idft(img_ishift, flags=cv2.DFT_SCALE)
    mag = cv2.magnitude(img_idft[:,:,0], img_idft[:,:,1])
    return mag.astype('uint8')

# apply median filter on bridge

med_fil_bridge = bridge.copy()
for t in range(50):
    med_fil_bridge = median_filter_on_freq(med_fil_bridge)

plt.figure(figsize=(12,16))
plt.subplot(1,2,1)
plt.title('Original Image')
plt.imshow(bridge, cmap='gray')
plt.subplot(1,2,2)
plt.title('Q3 (b) Image')
plt.imshow(med_fil_bridge, cmap='gray')
plt.show()

# apply median filter on fig_0514a

med_fil_0514 = fig_0514a.copy()
for t in range(50):
    med_fil_0514 = median_filter_on_freq(med_fil_0514)


plt.figure(figsize=(12,16))
plt.subplot(1,2,1)
plt.title('Original Image')
plt.imshow(fig_0514a, cmap='gray')
plt.subplot(1,2,2)
plt.title('Q3 (b) Image')
plt.imshow(med_fil_0514, cmap='gray')
plt.show()
