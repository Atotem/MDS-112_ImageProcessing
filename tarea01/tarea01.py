# -*- coding: utf-8 -*-
"""tarea01

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CpqBSafG6My7923PAx6l9mpVjzK1IBUi
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Ejercicio 1
img = cv2.imread('face.jpg') # Lee la imagen: debe ser el mismo nombre
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR to RGB

plt.imshow(img)
plt.show()

# Ejercicio 2
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # a continuacion se transforma la imagen a un canal en escala de grises

plt.imshow(img_gray, cmap='gray')
plt.show()

# Ejercicio 3
R, G, B = cv2.split(img) # En esta parte, la imagen es subdividida en sus 3 canales correspondientes

# Ejercicio 4
R_hist = cv2.calcHist([img], [0], None, [256], [0, 255]) # Se calcula el histograma del canal 'R', note que se ocupa la imagen original y el canal '0' (R de RGB)

plt.plot(R_hist, label='R_hist') # Se imprime el histograma del canal 'R'
# plt.show()

R_eq = cv2.equalizeHist(R) # Se ecualiza el canal 'R'
R_eq_hist = cv2.calcHist([R_eq], [0], None, [256], [0, 255]) # Se calcula el histograma del canalo 'R' ecualizado

plt.title('Histogramas Canal R')
plt.plot(R_eq_hist, label='R_eq_hist') # Se imprime el histograma del canal 'R' ecualizado
plt.legend()
plt.show()

plt.title('Imagen con Canal R Ecualizado')
plt.imshow(cv2.merge([R_eq, G, B])) # Se imprime la foto con todos sus canales, sin embargo, el canal 'R' esta ecualizado
plt.show() # Se imprimen todos los canales para apreciar de mejor manera el cambio

# Ejercicio 5

def correccion_gama(imagen, g): # Funcion Gama de la clase
  imagen = imagen/255 # Se divide la imagen para que se encuentre en el rango entre 0 y 1, es más eficiente el calculo de esta manera
  imagen = imagen**g # se eleva cada valor de la matriz por un factor gama 
  imagen = np.uint8(imagen*255) # se restaura los valores de la imagen a su escala original, y a enteros de 8 bits positivos
  return imagen # retorna el resultado

plt.title('Canal G')
plt.imshow(G, cmap='gray')
plt.show()

out1 = correccion_gama(G, 1.5)
plt.title('Canal G + Gamma 1.5')
plt.imshow(out1, cmap='gray')
plt.show()

out2 = correccion_gama(G, 0.4)
plt.title('Canal G + Gamma 0.4')
plt.imshow(out2, cmap='gray')
plt.show()

# Resultado final con la mejor correción gama segun el editor
plt.imshow(cv2.merge([R, out1, B]))
plt.show()

# Ejercicio 6

# max_frec = max(R_eq_hist)[0]
# bit_sig = list(R_eq_hist[:, 0]).index(max_frec)

# ret, BJ = cv2.threshold(img_gray, bit_sig, 255, cv2.THRESH_BINARY)

# plt.imshow(BJ, cmap="gray")
# plt.show()

for k in range(0, 8):
  plane = np.full((R_eq.shape[0], R_eq.shape[1]), 2**k, np.uint8) # Se obtiene cada plano a partir del canal 'R' ecualizado

  res = plane & img_gray # se multiplica por la imagen en escala de grises
  x = res*255 # se escala los valores de la imagen (matriz) a su escala original

  plt.imshow(x, cmap='gray')
  plt.show()