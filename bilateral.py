import cv2
import numpy as np
from matplotlib import pyplot as plt

# Este método é mais lento para calcular que os anteriores mas como vantagem
# apresenta a preservação de bordas e garante que o ruído seja removido

img = cv2.imread('clahe.jpg', 0)

# Tests

suave = cv2.bilateralFilter(img, 7, 49, 49)

cv2.imshow("7", suave)
cv2.waitKey(0)
cv2.imwrite("bilateral.png", suave)

#cv2.imshow("7", cv2.bilateralFilter(img, 7, 49, 49))
#cv2.waitKey(0)

#img = img[::2,::2] # Diminui a imagem

#suave = np.vstack([
 #np.hstack([img,
    #cv2.bilateralFilter(img, 3, 21, 21)]),
 #np.hstack([cv2.bilateralFilter(img, 5, 35, 35),
    #cv2.bilateralFilter(img, 7, 49, 49)]),
 #np.hstack([cv2.bilateralFilter(img, 9, 63, 63),
    #cv2.bilateralFilter(img, 11, 77, 77)])
#])

#cv2.imshow("Suavizada", suave)
#cv2.waitKey(0)
