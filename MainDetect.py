import DetectSkin
import DetectNail
from matplotlib.contour import ContourSet
from matplotlib.pyplot import contour

import numpy as np
import skimage.filters as sk_filters
from scipy import ndimage

# define the upper and lower boundaries of the HSV pixel
# intensities to be consired 'skin'
lower = np.array([8, 15, 110], dtype="uint8")
upper = np.array([30, 255, 255], dtype="uint8")


#filename = 'v1102.mp4'
#filename = 'v1104.mp4'
#filename = 'VRPS.mp4'
filename = 't8.mp4'
#filename = 'blackskin.wmv'


frame_number = 40

# Cria um retangulo que seleciona somente a max area da pele:
x , y, w, h = DetectSkin.DetectPositionMaxSkin(filename, frame_number, lower, upper)

# detecta a região da ponta do dedo:
DetectNail.DetectPositionMaxSkin(filename,x, y, w, h, lower, upper)

#cropeedIMAGE=DetectSkin.croppedSkin(filename,xc,yc,wc,hc)


#final_frame = cv2.hconcat((skin, skinMask))
#Show the concatenated frame using imshow.
#cv2.imshow('frame',skin)
