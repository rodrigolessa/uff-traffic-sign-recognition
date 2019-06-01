# v1.6 - Utilizando extração de contornos aplicado direto na imagem gray
#  e depois extraido o Zernike, ficou muito complexo e não produziu bons resultados.

# The main goal is to find the traffic sign in a photo and highlight it

# Import the necessary packages
from zernike_moments import ZernikeMoments
from searcher import Searcher
# import myutils as ut
# The image_utils contains convenience methods to handle basic image processing techniques
# resizing, rotating, and translating. 
import imutils
import numpy as np
#import argparse
import cv2
import pickle as cp
import matplotlib.pyplot as plt

def _img_resize(image, imSize):
	new = imutils.resize(image, height=imSize)
	if new.shape[1] > imSize:
		new = imutils.resize(image, width=imSize)

	border_size_x = (imSize - new.shape[1])//2
	border_size_y = (imSize - new.shape[0])//2

	new = cv2.copyMakeBorder(new, border_size_y + imSize, border_size_y + imSize, border_size_x + imSize, border_size_x + imSize, cv2.BORDER_REPLICATE)

	return new

# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-q", "--query", required = True, help = "Path to the query image")
# Only need one command line argument: 
# --query points to the path to where query image is stored on disk.
#args = vars(ap.parse_args())

# Tests signs:
##########################################################

imageName = "sinalizacao_brasileira_fotos\\edit\\permitido_ciclistas.jpg"
#  - similarity - v1

imageName = "sinalizacao_brasileira_fotos\\edit\\Passagem_sinalizada_de_pedestres_a.jpg"

imageName = "sinalizacao_brasileira_fotos\\edit\\Curva_acentuada_direita.jpg"

#imageName = "sinalizacao_brasileira_fotos\\edit\\De_a_preferencia.jpg"

# imageName = "sinalizacao_brasileira_fotos\\edit\\Parada_obrigatoria.jpg" 
# 0.03862711197553565 - similarity - v1

#imageName = "sinalizacao_brasileira_fotos\\edit\\Servicos_Auxiliares_i.png"
# 0.07263111311518465 - similarity - v1

#imageName = "sinalizacao_brasileira_fotos\\edit\\Velocidade_maxima_permitida.jpg"

#imageName = "sinalizacao_brasileira_fotos\\edit\\Proibido_transito_de_bicicletas_12.jpg"

imageRadius = 180

# Load the index of features
imageMomentsFile = open('index.pkl', 'rb')

indexa = cp.load(imageMomentsFile)

# Perform the search to identify the image
searcher = Searcher(indexa)

# Initialize descriptor with a radius of 180 pixels
zm = ZernikeMoments(180)

# Load the query image, 
image = cv2.imread(imageName)

# Only for the test image we have
# image = imutils.rotate(image, 90, center = None, scale = 1.0)

# Compute the ratio of the old height
# to the new height, clone it, and resize it
# ratio = image.shape[0] / 200.0

# Resize it - The smaller the image is, the faster it is to process
# image = cv2.resize(image, None, fx=0.95, fy=0.95, interpolation = cv2.INTER_CUBIC)
# image = ut.resize(image, width = 500, inter = cv2.INTER_CUBIC)
image = imutils.resize(image, height=500)

# Debugging: Show the original
cv2.imshow('original', image)
cv2.waitKey(0)

######################################################
# Separar os canais
# canais = cv2.split(image)
# cores = ('blue', 'green', 'red')
# b_channel, g_channel, r_channel = cv2.split(image)

# cv2.imshow("Canais", r_channel)
# cv2.waitKey(0)

# Convert the image to grayscale,
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# cv2.imshow("grayscale", r_channel)
# cv2.waitKey(0)

######################################################
# Equalização baseado em histograma da imagem
# CLAHE (Contrast Limited Adaptive Histogram Equalization)

# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

cl1 = clahe.apply(gray)

# res = np.hstack((img, cl1))
# cv2.imwrite('res.png', res)
# cv2.imshow("equalization", cl1)
# cv2.imwrite("equalization.jpg", cl1) # save frame as JPEG file
# cv2.waitKey(0)

# Blur the image slightly by using the cv2.bilateralFilter function
# Bilateral filtering has the nice property of removing noise in the image 
# while still preserving the actual edges.
blur = cv2.bilateralFilter(cl1, 9, 75, 75)

# Debugging:
cv2.imshow("blur", blur)
cv2.waitKey(0)

# Canny edge detector finds edge like regions in the image
# The Canny edge detector is an edge detection operator that uses 
# a multi-stage algorithm to detect a wide range of edges in images.
# It was developed by John F. Canny in 1986. 
# Canny also produced a computational theory of edge detection explaining why the technique works.
edged = cv2.Canny(blur, 30, 200)

# Debugging:
cv2.imshow("canny", edged)
# cv2.imwrite("canny.jpg", edged)
cv2.waitKey(0)

# Find contours in the edged image, keep only the largest ones, 
# and initialize our screen contour:
# The cv2.findContours function gives us a list of contours that it has found.
# The second parameter cv2.RETR_TREE tells OpenCV to compute the hierarchy 
# (relationship) between contours,
# We could have also used the cv2.RETR_LIST option as well;
# To compress the contours to save space using cv2.CV_CHAIN_APPROX_SIMPLE.
image2, cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.cv2.CV_CHAIN_APPROX_NONE)[1]

# Return only the 10 largest contours
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:15]

# Initialize empty list
lst_intensities = []

img = blur.copy()

# For each list of contour points...
# for i in range(len(cnts)):
        # Create a mask image that contains the contour filled in
        # cimg = np.zeros_like(img)
        # cv2.drawContours(cimg, cnts, i, color=255, thickness=-1)

        # Access the image pixels and create a 1D numpy array then add to list
        # pts = np.where(cimg == 255)
        # lst_intensities.append(img[pts[0], pts[1]])

# cv2.imshow("Threshold", cimg)
# cv2.waitKey(0)

# Initialize screenCnt, the contour that corresponds to our object to find
screenCnt = None

# Loop over contours
for c in cnts:
        # cv2.arcLength and cv2.approxPolyDP. 
        # These methods are used to approximate the polygonal curves of a contour.
        peri = cv2.arcLength(c, True)

        #print(peri)

        # Level of approximation precision. 
        # In this case, we use 2% of the perimeter of the contour.
        # * The Ramer–Douglas–Peucker algorithm, also known as the Douglas–Peucker algorithm and iterative end-point fit algorithm, 
        # is an algorithm that decimates a curve composed of line segments to a similar curve with fewer point
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # print(approx)

        # we know that a Object screen is a rectangle,
        # and we know that a rectangle has four sides, thus has four vertices.
        # If our approximated contour has four points, then
        # we can assume that we have found our screen.
        if len(approx) == 8 or len(approx) == 4 or len(approx) == 3:

                screenCnt = approx
                
                # Cortar a ára na imagem original
                x, y, w, h = cv2.boundingRect(approx)
                # make the box a little bigger
                x, y, w, h = x - 8, y - 8, w + 8, h + 8

                # rect = cv2.boundingRect(approx)
                # Mat subMat = new Mat(mRgba, rect)
                # Mat zeroMat = np.zeros(subMat.size(),subMat.type())
                # zeroMat.copyTo(subMat)

                # cv2.imshow("crop", crop)
                # cv2.waitKey(0)

                bouding = image.copy()
                
                # draw a green rectangle to visualize the bounding rect
                # cv2.drawContours(bouding, cv2.boundingRect(approx), -1, (0, 255, 0), 5)
                # cv2.rectangle(bouding, (x -8, y -8), (x + w + 8, y + h + 8), (0, 255, 0), 5)
                cv2.rectangle(bouding, (x, y), (x + w, y + h), (0, 255, 0), 5)
                
                # cv2.imshow("Boundingbox", bouding)
                # cv2.imwrite("boundingbox.jpg", bouding)
                # cv2.waitKey(0)

                crop = blur[y:y+h, x:x+w]

                # TODO: Exctract features
                # - Dominate color
                # - How many angles
                # - 

                # TODO: Clean image using mask of the perimeter

                # cv2.imshow("crop", crop)
                # cv2.waitKey(0)

                # Redimencionar imagem
                # resized = _img_resize(crop, width = 180, inter = cv2.INTER_CUBIC)
                # resized = _img_resize(blur, imageRadius)
                resized = imutils.resize(crop, width=180)

                # resized = add_border(resized, output_image='resized.jpg', border=100, color='white')
                row, col = resized.shape[:2]
                bottom = resized[row-2:row, 0:col]
                mean = cv2.mean(bottom)[0]

                bordersize = 100
                bordered = cv2.copyMakeBorder(resized, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[mean,mean,mean] )

                # mblur = cv2.medianBlur(crop, 5)
                # Binarização e limiar com Otsu
                _, thresh = cv2.threshold(bordered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                thresh = cv2.bitwise_not(thresh)
                # thresh[thresh > 0] = 255

                cv2.imshow("Threshold", thresh)
                # cv2.imwrite("threshold.jpg", thresh)
                cv2.waitKey(0)

                # Histograma de projeção/variância vertical
                # img_row_sum = np.sum(thresh, axis=1).tolist()
                # plt.plot(img_row_sum)
                # plt.show()
                # cv2.waitKey(0)

                # Drawing our screen contours, we can clearly see that we have found the Object screen
                # if isinstance(screenCnt, list):
                cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 5)

                cv2.imshow("object", image)
                # cv2.imwrite("object.jpg", image)
                cv2.waitKey(0)

                # res = np.hstack((original, edged, image))

                # Extrair a característica:
                # Encontrar o histograma de cada camada
                # Encontrar a área
                # Encontrar a Variância de projeção vertical
		# Compute Zernike moments to characterize the shape of object outline
                moments = zm.describe(thresh)

                # print(moments)

                # Utilizar o dataset anotado de placas e comparar utilizando distância Euclidiana
                # Return 5 first similarities
                results = searcher.search(indexa, moments)[:10]
                for r in results:
                        image_ref = r[1]
                        image_distance = r[0]

                        if image_distance <= 0.109:
                                print("Objeto semelhante: {} - {}".format(image_distance, image_ref))

                break

cv2.destroyAllWindows()