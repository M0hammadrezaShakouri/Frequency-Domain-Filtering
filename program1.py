import logging
import numpy as np
import cv2


def butterworth_filter(I_shape):
    P = I_shape[0]/2
    Q = I_shape[1]/2
    U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
    Duv = (((U-P)**2+(V-Q)**2)).astype(float)
    H = 1/(1+(Duv/30**2)**2)
    return (1 - H)

def gaussian_filter(I_shape):
    P = I_shape[0]/2
    Q = I_shape[1]/2
    H = np.zeros(I_shape)
    U, V = np.meshgrid(range(I_shape[0]), range(I_shape[1]), sparse=False, indexing='ij')
    Duv = (((U-P)**2+(V-Q)**2)).astype(float)
    H = np.exp((-Duv/(2*(30)**2)))
    return (1 - H)

path1 = "monkey.png"
path2 = "cameraman.jpg"

def zarb(x,H):
    H = np.fft.fftshift(H)
    I_filtered = (0.70+ 1.25*H)*x
    return I_filtered

def filter(x,filtername=None):
    I_log = np.log1p(np.array(x, dtype="float"))
    I_fft = np.fft.fft2(I_log)

    if filtername=='butterworth':
        H = butterworth_filter(I_shape = I_fft.shape)
    elif filtername=='gaussian':
        H = gaussian_filter(I_shape = I_fft.shape)
    
    # Apply filter on frequency domain then take the image back to spatial domain
    I_fft_filt = zarb(I_fft, H)
    I_filt = np.fft.ifft2(I_fft_filt)
    I = np.exp(np.real(I_filt))-1
    return np.uint8(I)

if __name__ == "__main__":
    
    x = cv2.imread(path2)[:, :, 0]
    y = filter(x,"butterworth")
    cv2.imwrite("homo-butter.jpg", y)
    y = filter(x,"gaussian")
    cv2.imwrite("homo-gaussian.jpg", y)
