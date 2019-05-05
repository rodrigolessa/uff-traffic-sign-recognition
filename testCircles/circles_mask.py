# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghcircles/py_houghcircles.html
import cv2
import numpy as np

# load the image
#img = cv2.imread('opencv_logo.png')
#img = cv2.imread('bicicletas.jpg')
#img = cv2.imread('proibido_bicicletas.jpg')
img = cv2.imread('preferencia.jpg')
img = cv2.imread('curva.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# detect circles
gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 5)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=50, minRadius=0, maxRadius=0)
circles = np.uint16(np.around(circles))

print(circles)

# draw mask
mask = np.full((img.shape[0], img.shape[1]), 0, dtype=np.uint8)  # mask is only 
for i in circles[0, :]:
    cv2.circle(mask, (i[0], i[1]), i[2], (255, 255, 255), -1)
cv2.imshow("mask", mask)
cv2.waitKey(0)

# get first masked value (foreground)
fg = cv2.bitwise_or(img, img, mask=mask)
cv2.imshow("masked", fg)
cv2.waitKey(0)

# get second masked value (background) mask must be inverted
mask = cv2.bitwise_not(mask)
background = np.full(img.shape, 255, dtype=np.uint8)
bk = cv2.bitwise_or(background, background, mask=mask)

# combine foreground+background
final = cv2.bitwise_or(fg, bk)

cv2.imshow("final", final)
cv2.waitKey(0)
cv2.destroyAllWindows()