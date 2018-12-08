import numpy as np
import cv2

img = cv2.imread('sinalizacao_brasileira_fotos\\edit\\152116938.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

(canalAzul, canalVerde, canalVermelho) = cv2.split(img)

zeros = np.zeros(img.shape[:2], dtype = "uint8")

cv2.imshow("Vermelho", cv2.merge([zeros, zeros, canalVermelho]))
cv2.imshow("Verde", cv2.merge([zeros, canalVerde, zeros]))
cv2.imshow("Azul", cv2.merge([canalAzul, zeros, zeros]))
cv2.imshow("Original", img)
cv2.imshow("Grayscale", gray)
cv2.waitKey(0)

cv2.imwrite("grayscale.jpg", gray)