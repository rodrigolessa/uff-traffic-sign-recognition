# The main goal is to find the screen of Game and highlight it, 

# iIport the necessary packages
# The image_utils contains convenience methods to handle basic image processing techniques
# resizing, rotating, and translating. 
import imutils
import numpy as np
#import argparse
import cv2
import matplotlib.pyplot as plt
#from skimage import exposure
#from skimage import data, io, filters
#image = data.coins()
## ... or any other NumPy array!
#edges = filters.sobel(image)
#io.imshow(edges)
#io.show()

# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-q", "--query", required = True, help = "Path to the query image")
# Only need one command line argument: 
# --query points to the path to where query image is stored on disk.
#args = vars(ap.parse_args())

# Triangle signs:
#imageName = "dataset_traffic_sign/a100036_traffic-sign_7-2.jpg"
#imageName = "dataset_traffic_sign/480px_thailand_road.png"
#imageName = "dataset_traffic_sign/ANZ_traffic_lights_ahead_sign.png"
imageName = "sinalizacao_brasileira_fotos\\edit\\152116938.jpg"
imageNumber = "4"
imageRef = "traffic_sign_canny_square_"

# Load the query image, 
image = cv2.imread(imageName)

# Only for the test image we have
# image = imutils.rotate(image, 90, center = None, scale = 1.0)

# compute the ratio of the old height
# to the new height, clone it, and resize it
ratio = image.shape[0] / 200.0

# resize it - The smaller the image is, the faster it is to process
#image = imutils.resize(image, height = 900)

image = cv2.resize(image, None, fx=0.95, fy=0.95, interpolation = cv2.INTER_CUBIC)

# Debugging: Show Original
cv2.imshow('original', image)
cv2.waitKey(0)

# clone it
#original = image.copy()
#cv2.imwrite(imageRef + imageNumber + "1.jpg", image)

#print(imageRef + imageNumber + "1.jpg")

# Separar os canais
#canais = cv2.split(image)
#cores = ('blue', 'green', 'red')
b_channel, g_channel, r_channel = cv2.split(image)

cv2.imshow("Canais", r_channel)
#cv2.imwrite("eualization.jpg", cl1) # save frame as JPEG file
cv2.waitKey(0)

# Convert the image to grayscale, blur it, 
# and find edges in the image
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

######################################################
# Equalização baseado em histograma da imagem

#CLAHE (Contrast Limited Adaptive Histogram Equalization)

# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#clahe = cv2.createCLAHE()

cl1 = clahe.apply(r_channel)

#res = np.hstack((img, cl1))
#cv2.imwrite('res.png', res)
cv2.imshow("Equalization", cl1)
#cv2.imwrite("eualization.jpg", cl1) # save frame as JPEG file
cv2.waitKey(0)

# Blur the image slightly by using the cv2.bilateralFilter function
# Bilateral filtering has the nice property of removing noise in the image 
# while still preserving the actual edges.
blur = cv2.bilateralFilter(cl1, 11, 17, 17)

# Debugging:
cv2.imshow("Blur", blur)
cv2.waitKey(0)

# Canny edge detector finds edge like regions in the image
#* The Canny edge detector is an edge detection operator that uses a multi-stage algorithm to detect a wide range of edges in images. 
# It was developed by John F. Canny in 1986. Canny also produced a computational theory of edge detection explaining why the technique works.
edged = cv2.Canny(blur, 30, 200)

# Debugging:
cv2.imshow("Canny", edged)
cv2.imwrite("canny.jpg", edged)
cv2.waitKey(0)

#cv2.imwrite(imageRef + imageNumber + "2.jpg", edged)

# Find contours in the edged image, keep only the largest ones, 
# and initialize our screen contour:
# The cv2.findContours function gives us a list of contours that it has found.
# The second parameter cv2.RETR_TREE tells OpenCV to compute the hierarchy 
# (relationship) between contours,
# We could have also used the cv2.RETR_LIST option as well;
# To compress the contours to save space using cv2.CV_CHAIN_APPROX_SIMPLE.
image2, cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Return only the 10 largest contours
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:100]

# Initialize screenCnt, the contour that corresponds to our object to find
screenCnt = None

# Loop over contours
for c in cnts:
    # cv2.arcLength and cv2.approxPolyDP. 
    # These methods are used to approximate the polygonal curves of a contour.
    peri = cv2.arcLength(c, True)

    print(peri)

    # Level of approximation precision. 
    # In this case, we use 2% of the perimeter of the contour.
    #* The Ramer–Douglas–Peucker algorithm, also known as the Douglas–Peucker algorithm and iterative end-point fit algorithm, 
    # is an algorithm that decimates a curve composed of line segments to a similar curve with fewer point
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    print(approx)

    # we know that a Object screen is a rectangle,
    # and we know that a rectangle has four sides, thus has four vertices.
    # If our approximated contour has four points, then
    # we can assume that we have found our screen.
    if len(approx) == 4:
        screenCnt = approx
        # Cortar a ára na imagem original
        x, y, w, h = cv2.boundingRect(approx)
        # make the box a little bigger
        #x, y, w, h = x-5, y-5, w+5, h+5
        crop = blur[y:y+h, x:x+w]
        cv2.imshow("Crop the blur thing", crop)
        cv2.waitKey(0)

        #mblur = cv2.medianBlur(crop, 5)
        _, thresh = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh = cv2.bitwise_not(thresh)
        #thresh[thresh > 0] = 255

        cv2.imshow("Crop the original", thresh)
        cv2.imwrite("thresh.jpg", thresh)
        cv2.waitKey(0)

        img_row_sum = np.sum(thresh, axis=1).tolist()
        plt.plot(img_row_sum)
        plt.show()
        cv2.waitKey(0)
        # Extrair a característica:
        # Encontrar o histograma de cada camada
        # Binarização e limiar com Otsu
        # Extrair as características:
        # Encontrar a área
        # Encontrar a Variância de projeção vertical
        # Utilizar o dataset anotado de placas e comparar utilizando distância Euclidiana
        break

# Drawing our screen contours, we can clearly see that we have found the Object screen
#if isinstance(screenCnt, list):
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 5)
cv2.imshow("Object Screen", image)
cv2.imwrite("object.jpg", image)
cv2.waitKey(0)

#res = np.hstack((original, edged, image))

#cv2.imwrite('traffic_sign_canny_douglaspeucker_1.jpg', res)
#cv2.imshow("Resultados", res)
#cv2.waitKey(0)

cv2.destroyAllWindows()