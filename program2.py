
import cv2 as cv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import copy


path1 = "monkey.png"
path2 = "cameraman.jpg"

def get_image(path=None):
    x = cv.imread(path,0)
    return x

def show_image(x,st="QWERT"):
    newpath = st
    cv.imwrite(newpath, x)     
    plt.title(newpath)         
    plt.imshow(x,cmap="gray")
    plt.show()


def padding(img):
    H, W = img.shape
    
    zimg = np.zeros((H+4,W+4))
    zimg[2:H+2,2:W+2] = img
    
    zimg[2:H+2,0:2]=img[:,0:1]
    zimg[H+2:H+4,2:W+2]=img[H-1:H,:]
    zimg[2:H+2,W+2:W+4]=img[:,W-1:W]
    zimg[0:2,2:W+2]=img[0:1,:]
    
    zimg[0:2,0:2]=img[0,0]
    zimg[H+2:H+4]=img[H-1,0]
    zimg[H+2:H+4,W+2:W+4]=img[H-1,W-1]
    zimg[0:2,W+2:W+4]=img[0,W-1]
    return zimg
    
def show_magnitude(x,ft):
    magnitude_spectrum = 20*np.log(np.abs(ft))
    plt.subplot(121),plt.imshow(x, cmap = 'gray')
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.show()

def dft(x):

    f = np.fft.fft2(x)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    print(fshift.shape)
    
    show_magnitude(x,fshift)
    return fshift

def idft(fx):
    f_ishift = np.fft.ifftshift(fx)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back

def mask(x):
    rows, cols = x.shape
    crow,ccol = int(rows/2) , int(cols/2)
    mask = np.zeros((rows,cols,2),np.uint8)
    mask[crow-30:crow+30, ccol-30:ccol+30] = 1

def mask(x, fx,name="None"):
    if(name == "bala_gozar_ideal"):
        rows, cols = x.shape
        crow,ccol = int(rows/2) , int(cols/2)
        thresh = 10
        fx[crow-thresh:crow+thresh, ccol-thresh:ccol+thresh] = 0
    if(name == "payin_gozar_ideal"):
        rows, cols = x.shape
        crow,ccol = int(rows/2) , int(cols/2)
        thresh = 100
        fx[:crow-thresh, :] = 0
        fx[:, ccol+thresh:] = 0
        fx[crow+thresh:, :] = 0
        fx[:, :ccol-thresh] = 0
    if(name == "mian_gozar_ideal"):
        rows, cols = x.shape
        crow,ccol = int(rows/2) , int(cols/2)
        thresh1 = 10
        fx[crow-thresh1:crow+thresh1, ccol-thresh1:ccol+thresh1] = 0
        thresh = 100
        fx[:crow-thresh, :] = 0
        fx[:, ccol+thresh:] = 0
        fx[crow+thresh:, :] = 0
        fx[:, :ccol-thresh] = 0

    return fx

if __name__ == "__main__":

    x = get_image(path1)
    x = cv2.resize(x, (250,250), interpolation = cv2.INTER_AREA)

    fx = dft(x)
    fx_mask = mask(x,fx, "payin_gozar_ideal")
    ffx = idft(fx_mask)
    print(ffx.shape)
    plt.imshow(ffx, cmap = 'gray')
    plt.show()
    
