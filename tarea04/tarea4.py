# -*- coding: utf-8 -*-
"""tarea4

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10ohQkB6L-OBc_TR19SGsOwP38wTM74ed
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2

img = cv2.imread(rombo.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray_norm = cv2.normalize(gray.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
noise = np.random.random(gray.shape)*0.3
output = gray_norm + noise