import cv2
import numpy as np
import matplotlib.pyplot as plt

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
