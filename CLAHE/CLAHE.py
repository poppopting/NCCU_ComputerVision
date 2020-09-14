#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2


# In[2]:


class My_Clahe():
    def __init__(self, clipLimit=0.5, tileGridSize=(8, 8), L=256):
        
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize
        self.ker_v, self.ker_h = tileGridSize
        self.L = L
        
    def _symetric_pad(self, img):
        
        y, x = img.shape
        # compute number of pixels to pad 
        pad_v = (self.ker_v - (y % self.ker_v)) % self.ker_v
        pad_h = (self.ker_h - (x % self.ker_h)) % self.ker_h
        # pad only on right side and bottom side
        pad_img = np.pad(img, ((0,pad_v), (0,pad_h)), 'symmetric')
        
        return pad_img
    
    def _clip_and_hist(self, SubImg):
        
        img_flatten = SubImg.flatten()
        hist, _ = np.histogram(img_flatten, bins=np.arange(self.L+1))
        total = sum(hist)
        # compute upper bound of histogram 
        hist_bound = self.clipLimit*total
        while np.max(hist) > hist_bound :
            
            hist = hist.clip(max=hist_bound)
            # compute number of cliped piexl and how much increment should we add
            increment = (total - sum(hist)) / self.L
            hist += increment
        # ordinary HE
        pdf = hist / total
        cdf = pdf.cumsum()
        transform = np.ceil((self.L-1) * cdf).astype('uint8')

        return transform
    
    def _bilinear_interpolation(self, DeltaX, DeltaY, UL, BL, UR, BR):

        return (1-DeltaY)*((1-DeltaX)*UL + DeltaX*UR) + DeltaY*((1-DeltaX)*BL + DeltaX*BR)

    def _linear_interpolation(self, Delta, U_L, B_R):
        
        return (1-Delta)*U_L + Delta*B_R
    
    def apply(self, img):
        # pad first
        pad_img = self._symetric_pad(img)
        pad_y, pad_x = pad_img.shape
        
        #construct array to store pixel after clahe
        clahe_img = np.zeros(img.shape)
        #consturct array to store tramform function of each sub image
        trans_map = np.zeros((int(pad_y/self.ker_v), int(pad_x/self.ker_h), self.L))
        # HE each sub image and store tramform functions
        for v in range(0, pad_y, self.ker_v):
            for h in range(0, pad_x, self.ker_h):
                SubImg = pad_img[v:v+self.ker_v, h:h+self.ker_h]
                trans_map[int(v/self.ker_v), int(h/self.ker_h), :] = self._clip_and_hist(SubImg)
                
        # begin interpolation
        y, x = img.shape
        for v in range(y):
            for h in range(x):
                v_lower = self.ker_v / 2
                v_upper = pad_y - self.ker_v/2 - 1
                h_lower = self.ker_h / 2
                h_upper = pad_x - self.ker_h/2 - 1
                
                pixel = img[v,h]
                ### CORNER : directly map by tranform function of nearest sub image 
                # Upper-Left
                if (v < v_lower) and (h < h_lower) :
                    clahe_img[v,h] = trans_map[0,0,pixel]
                # Bottom-Left
                elif (v > v_upper) and (h < h_lower) :
                    clahe_img[v,h] = trans_map[-1,0,pixel]
                # Upper-Right
                elif (v < v_lower) and (h > h_upper) :
                    clahe_img[v,h] = trans_map[0,-1,pixel]
                # Bottom-Right
                elif (v > v_upper) and (h > h_upper) :
                    clahe_img[v,h] = trans_map[-1,-1,pixel]

                ### BORDER : linear interpolation by tranform functions of nearest 2 sub images
                # maybe up and down or left and right
                # Left
                elif (v_lower <= v <= v_upper) and (h < h_lower) :
                    div, mod = divmod(v - int(self.ker_v/2), self.ker_v)
                    # compute distance from pixel to center point
                    Delta = mod + 0.5 
                    Delta /= self.ker_v
                    # div : which tranform function to use    
                    U_L = trans_map[div,0,pixel]
                    B_R = trans_map[div+1,0,pixel]
       
                    clahe_img[v,h] = self._linear_interpolation(Delta, U_L, B_R)
     
                # Right
                elif (v_lower <= v <= v_upper) and (h > h_upper) :
                    div, mod = divmod(v - int(self.ker_v/2), self.ker_v)
                    # compute distance from pixel to center point
                    Delta = mod + 0.5 
                    Delta /= self.ker_v
                    # div : which tranform function to use   
                    U_L = trans_map[div,-1,pixel]
                    B_R = trans_map[div+1,-1,pixel]
      
                    clahe_img[v,h] = self._linear_interpolation(Delta, U_L, B_R)
       
                # Up
                elif (v < v_lower) and (h_lower <= h <= h_upper) :
                    div, mod = divmod(h - int(self.ker_h/2), self.ker_h)
                    # compute distance from pixel to center point
                    Delta = mod + 0.5 
                    Delta /= self.ker_h
                    # div : which tranform function to use 
                    U_L = trans_map[0,div,pixel]
                    B_R = trans_map[0,div+1,pixel]
                    
                    clahe_img[v,h] = self._linear_interpolation(Delta, U_L, B_R)

                # Bottom
                elif (v > v_upper) and (h_lower <= h <= h_upper) :
                    div, mod = divmod(h - int(self.ker_h/2), self.ker_h)
                    # compute distance from pixel to center point
                    Delta = mod + 0.5 
                    Delta /= self.ker_h
                    # div : which tranform function to use 
                    U_L = trans_map[-1,div,pixel]
                    B_R = trans_map[-1,div+1,pixel]
                    
                    clahe_img[v,h] = self._linear_interpolation(Delta, U_L, B_R)
     
                ### INTERNAL PIXEL: bilinear interpolation by tranform functions of nearest 4 sub images
                else :
                    # compute distance from pixel to center point
                    div_v, mod_v = divmod(v - int(self.ker_v/2), self.ker_v)
                    div_h, mod_h = divmod(h - int(self.ker_h/2), self.ker_h)
                    
                    DeltaX = mod_h + 0.5 
                    DeltaY = mod_v + 0.5 
                    DeltaX /= self.ker_h
                    DeltaY /= self.ker_v
					# div_v : which tranform function to use on vertical line
					# div_h : which tranform function to use on horizional line
                    UL = trans_map[div_v,div_h,pixel]
                    BL = trans_map[div_v+1,div_h,pixel]
                    UR = trans_map[div_v,div_h+1,pixel]
                    BR = trans_map[div_v+1,div_h+1,pixel]

                    clahe_img[v,h] = self._bilinear_interpolation(DeltaX, DeltaY, UL, BL, UR, BR)
             
        return clahe_img.astype('uint8')


# In[3]:


mp2 = cv2.imread('mp2.jpg', cv2.IMREAD_GRAYSCALE)


# In[8]:


# my clahe
MyClahe = My_Clahe(clipLimit=.015)
clahe_img = MyClahe.apply(mp2)
cv2.imwrite('my_clahe.jpg', clahe_img)


# In[5]:


# clahe opencv
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_cv = clahe.apply(mp2)
cv2.imwrite('cv_clahe.jpg', clahe_cv)

