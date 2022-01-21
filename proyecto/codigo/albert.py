# -*- coding: utf-8 -*-
"""Albert

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1N0oTZ6tRIKQb_rYyu067U1ua1B0xqOuO

# Codigo ejemplo
## Autor: Alberto Bella
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.filters import threshold_otsu
from skimage import measure
import scipy.ndimage as ndi
import dlib

# def drawRectangle(n_img, x, y, height, width, r):
#     # img must be a 3-dimensional array and in RGB order
#     # x, y positions belong to left top corner
#     # height and width of the square
#     # r is the border weight. Must be greater than 0

#     if r<=0:
#         print('r must be greater than 0')
#         return 0

#     # right side
#     n_img[y:y+height, x:x+r, :] = 0
#     n_img[y:y+height, x:x+r, 0] = 255

#     # left side
#     n_img[y:y+height, x+width:x+width+r, :] = 0
#     n_img[y:y+height, x+width:x+width+r, 0] = 255

#     # top side
#     n_img[y:y+r, x:x+width, :] = 0
#     n_img[y:y+r, x:x+width, 0] = 255

#     # bottom side
#     n_img[y+height:y+height+r, x:x+width+r, :] = 0
#     n_img[y+height:y+height+r, x:x+width+r, 0] = 255

img = cv2.imread('fotos/Caras01.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

imgs = [img_rgb, img_hsv, img_ycrcb]
titles = ['img_rgb', 'img_hsv', 'img_ycrcb']
color = ['red', 'green', 'blue']
channels = [['r', 'g', 'b'], ['h', 's', 'v'], ['y', 'cr', 'cb']]

# hack
detector = dlib.get_frontal_face_detector()
dets = detector(img, 1)

# plot hack
# img_n = img_rgb
# dets = detector(img_n, 1)
# for det in range(len(dets)):
#   drawRectangle(img_n, dets[det].left(), dets[det].top(), dets[det].height(), dets[det].width(), 10)
# plt.imshow(img_n)
# plt.show()

# plot face
face = 1
face0 = img_rgb[dets[face].top():dets[face].bottom(), dets[face].left():dets[face].right()]
plt.title('face0')
plt.imshow(face0)
plt.show()

for i in range(0, len(imgs)):
  face0 = imgs[i][dets[face].top():dets[face].bottom(), dets[face].left():dets[face].right()]
  # plot hist for each image
  legend = []
  for channel in range(3):
    # add each channel to hist plot
    hst = cv2.calcHist([face0], [channel], None, [256], [0, 255])
    plt.title('face0 \ ' + titles[i])
    plt.plot(hst, c=color[channel])
    legend.append(channels[i][channel])
  plt.legend(legend)
  plt.show()

# plot official image
plt.title('img_rgb')
plt.imshow(img_rgb)
plt.show()

for i in range(0, len(imgs)):
  for channel in range(imgs[i].shape[2]):
    hst = cv2.calcHist([imgs[i]], [channel], None, [256], [0, 255])
    plt.title(titles[i])
    plt.plot(hst, c=color[channel])
  plt.show()

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

hst = cv2.calcHist(img_gray, [0], None, [256], [0, 255])
plt.title('img_gray')
plt.plot(hst)
plt.show()

# global thresholding
ret1, th1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
plt.title('global thresholding')
plt.imshow(th1, cmap='gray')
plt.show()

# Otsu's thresholding
ret2, th2 = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.title('Otsu´s thresholding')
plt.imshow(th2, cmap='gray')
plt.show()

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img_gray, (5,5), 0)
ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.title('Otsu´s thresholding after Gaussian filtering')
plt.imshow(th3, cmap='gray')
plt.show()

blur = cv2.medianBlur(img_gray, 5)

# binary
ret4, th4 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
plt.title('binary')
plt.imshow(th4, cmap='gray')
plt.show()

# adaptive binary mean
th5 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
plt.title('adaptive binary mean')
plt.imshow(th5, cmap='gray')
plt.show()

# adaptive binary gaussian
th6 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
plt.title('adaptive binary gaussian')
plt.imshow(th6, cmap='gray')
plt.show()

# ejemplo modelo

def filtro_geometrico(A):
    largo =  len(A)
    S = np.prod(A.flatten())**(1/largo)
    return S

# apply adaptive gaussian threshold
ret, thresh = cv2.threshold(img_hsv[:, :, 0], 12, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C) # values are between 0 and 255
    
# aplicamos el filtro promedio alfa-acotado
filtro = ndi.generic_filter(thresh, filtro_geometrico, [5,5])

# apply erosion
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
erode = cv2.erode(filtro, kernel)

result = erode

# evaluation
# read eval image
refer = cv2.imread('fotos/Refer01.bmp')
refer = cv2.cvtColor(refer, cv2.COLOR_BGR2GRAY)

# transformar a una imagen binaria
img_bin_ideal = refer > 127 # must be 0 or 255
img_ideal_segmentada = result > 127 # must be 0 or 255

# dejamos valores en ceros y unos
img_bin_ideal = img_bin_ideal*1
img_ideal_segmentada = img_ideal_segmentada*1

# calcular diferencias
diferencias = img_bin_ideal -  img_ideal_segmentada

# calcular estadisticas
TP = np.sum(diferencias == 0)
FP = np.sum(diferencias == -1)
FN = np.sum(diferencias == 1)

print('Hay {} True Positive'.format(TP))
print('Hay {} False Positive'.format(FP))
print('Hay {} False Negative'.format(FN))

# despliegue de imagenes
fig, ax = plt.subplots(nrows=1, ncols=3, dpi=150)
ax[0].imshow(img_bin_ideal, cmap='gray')
ax[1].imshow(img_ideal_segmentada, cmap='gray')
ax[2].imshow(diferencias, cmap='gray')

ax[0].set_title('Imagen ideal')
ax[1].set_title('Imagen Segmentada')
ax[2].set_title('Diferencias')
plt.show()

TPR = TP / (TP + FN)
P = TP / (FP + TP)
F1_score = 2 * (TPR * P) / (TPR + P)

print('F1 score: {}'.format(round(F1_score, 5)))