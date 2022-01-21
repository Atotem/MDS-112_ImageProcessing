import cv2
import matplotlib.pyplot as plt
import numpy as np

def segmantacion(imagen):
    img = cv2.imread(imagen)
    img_HSI = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    img_YCrCb = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)

    dst = cv2.equalizeHist(img_HSI[:,:,0])

    # plt.title('Imagen Ecualizada')
    # plt.imshow(dst, cmap="gray")
    # plt.show()

    ret, Bin1= cv2.threshold(dst, 30, 255, type=cv2.ADAPTIVE_THRESH_GAUSSIAN_C)

    # plt.title('Imagen Threshold')
    # plt.imshow(Bin1, cmap="gray")
    # plt.show()

    kernel_erosion = cv2.getStructuringElement(cv2.MORPH_CROSS, (15,15))
    kernel_dilatacion = cv2.getStructuringElement(cv2.MORPH_CROSS, (28,28))

    img_erosionada = cv2.erode(Bin1, kernel_erosion)

    # plt.title('Imagen erosionada')
    # plt.imshow(img_erosionada, cmap="gray")
    # plt.show()

    img_dilatada = cv2.dilate(img_erosionada, kernel_dilatacion)

    # plt.title('Imagen Apertura')
    # plt.imshow(img_dilatada, cmap="gray")
    # plt.show()

    img_caras = img_dilatada
    img_caras[700:, :] = 0

    # plt.title('Imagen Apertura caras')
    # plt.imshow(img_caras, cmap="gray")
    # plt.show()

    return img_caras

imagenes = ['Caras01.jpg', 'Caras02.jpg', 'Caras03.jpg', 'Caras04.jpg', 'Caras05.jpg', 'Caras06.jpg', 'Caras07.jpg', ]

for i in imagenes:
    plt.imshow(segmantacion(i), cmap="gray")
    plt.title(i+' Segmentada')
    plt.savefig(i[0:7]+'_segmentada.jpg')
    plt.show()