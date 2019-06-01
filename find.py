# The main goal is to find the traffic sign in a photo and highlight it

# Import the necessary packages
from zernike_moments import ZernikeMoments
from searcher import Searcher
# import myutils as ut
# The img_utils contains convenience methods to handle basic img processing techniques
# resizing, rotating, and translating. 
import imutils
import numpy as np
#import argparse
import cv2
import pickle as cp
import matplotlib.pyplot as plt

def _img_resize(img, imSize):
	new = imutils.resize(img, height=imSize)
	if new.shape[1] > imSize:
		new = imutils.resize(img, width=imSize)

	border_size_x = (imSize - new.shape[1])//2
	border_size_y = (imSize - new.shape[0])//2

	new = cv2.copyMakeBorder(new, border_size_y + imSize, border_size_y + imSize, border_size_x + imSize, border_size_x + imSize, cv2.BORDER_REPLICATE)

	return new

# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-q", "--query", required = True, help = "Path to the query img")
# Only need one command line argument: 
# --query points to the path to where query img is stored on disk.
#args = vars(ap.parse_args())

######################################################
# Parameters:

#imgName = "sinalizacao_brasileira_fotos\\edit\\permitido_ciclistas.jpg"
#imgName = "sinalizacao_brasileira_fotos\\edit\\Passagem_sinalizada_de_pedestres_a.jpg"
imgName = "sinalizacao_brasileira_fotos\\edit\\Curva_acentuada_direita.jpg"
#imgName = "sinalizacao_brasileira_fotos\\edit\\De_a_preferencia.jpg"
# imgName = "sinalizacao_brasileira_fotos\\edit\\Parada_obrigatoria.jpg" 
# 0.03862711197553565 - similarity - v1
#imgName = "sinalizacao_brasileira_fotos\\edit\\Servicos_Auxiliares_i.png"
# 0.07263111311518465 - similarity - v1
#imgName = "sinalizacao_brasileira_fotos\\edit\\Velocidade_maxima_permitida.jpg"
#imgName = "sinalizacao_brasileira_fotos\\edit\\Proibido_transito_de_bicicletas_12.jpg"

imgMomentsFileName = 'index.pkl'

imgRadius = 180

# Load the index of features
imgMomentsFile = open(imgMomentsFileName, 'rb')

indexA = cp.load(imgMomentsFile)

# Perform the search to identify the img
searcher = Searcher(indexA)

# Initialize descriptor with a radius of 180 pixels
zm = ZernikeMoments(imgRadius)

######################################################
# Load the query img, 
img = cv2.imread(imgName)

# Resize it - The smaller the img is, the faster it is to process
# img = cv2.resize(img, None, fx=0.95, fy=0.95, interpolation = cv2.INTER_CUBIC)
# img = ut.resize(img, width = 500, inter = cv2.INTER_CUBIC)
img = imutils.resize(img, height=500)

# Debugging: Show the original img
cv2.imshow('original', img)
#cv2.waitKey(0)

######################################################
# Separar os canais

# TODO: Levar para outro espaço de cor: de RGB -> HSV
# HSV - Sistemas de cores formado pelos componentes:
# - Matiz (hue)
#       Representa todas as cores puras. 
#       Geralmente é representada por ângulos, 
#       começa em 0 no vermelhor e termina em 360 também no vermelhor
# - Saturação (saturation) 
#       O quanto a cor possui o componente de cor branca
# - Brilho (value) 
#       Noção acromática de intensidade

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

hsvWhite = np.asarray([0, 0, 255])    # white!
hsvYellow = np.asarray([40, 255, 255]) # yellow! note the order

mask = cv2.inRange(hsv, hsvWhite, hsvYellow)
#print(mask)


# Bitwise-AND mask and original image
res = cv2.bitwise_and(hsv, hsv, mask=mask)
ratio = cv2.countNonZero(mask)/(hsv.size/3)
print('pixel percentage:', np.round(ratio*100, 2))
#plt.imshow(mask)

cv2.imshow('mask',res)

#plt.imshow(mask, cmap='gray')   # this colormap will display in black / white
#plt.show()
cv2.waitKey(0)


(canalAzul, canalVerde, canalVermelho) = cv2.split(hsv)

zeros = np.zeros(img.shape[:2], dtype = "uint8")

cv2.imshow("Vermelho", cv2.merge([zeros, zeros, canalVermelho]))
cv2.imshow("Verde", cv2.merge([zeros, canalVerde, zeros]))
cv2.imshow("Azul", cv2.merge([canalAzul, zeros, zeros]))
cv2.waitKey(0)

# TODO: Obter as cores: Vermelhor, amarelo e azul
# TODO: Obter a saturação em uma intensidade alta

# cv2.imshow("grayscale", r_channel)
# cv2.waitKey(0)

######################################################
# Equalização baseado em histograma da imgm
# CLAHE (Contrast Limited Adaptive Histogram Equalization)

# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

cl1 = clahe.apply(gray)

# res = np.hstack((img, cl1))
# cv2.imwrite('res.png', res)
# cv2.imshow("equalization", cl1)
# cv2.imwrite("equalization.jpg", cl1) # save frame as JPEG file
# cv2.waitKey(0)

# Blur the img slightly by using the cv2.bilateralFilter function
# Bilateral filtering has the nice property of removing noise in the img 
# while still preserving the actual edges.
blur = cv2.bilateralFilter(cl1, 9, 75, 75)

# TODO: Usando segmentação por cor

# Debugging:
cv2.imshow("blur", blur)
cv2.waitKey(0)

# TODO: Aplicar filtro Canny (#cani) ou outro mais simples para extrair contornos

# Canny edge detector finds edge like regions in the img
# The Canny edge detector is an edge detection operator that uses 
# a multi-stage algorithm to detect a wide range of edges in imgs.
# It was developed by John F. Canny in 1986. 
# Canny also produced a computational theory of edge detection explaining why the technique works.
edged = cv2.Canny(blur, 30, 200)

# Debugging:
cv2.imshow("canny", edged)
# cv2.imwrite("canny.jpg", edged)
cv2.waitKey(0)

# TODO: Aplicar filtros morfológicos

# TODO: Aplicar a transformada de Hough (#Rufi)

# Find contours in the edged img, keep only the largest ones, 
# and initialize our screen contour:
# The cv2.findContours function gives us a list of contours that it has found.
# The second parameter cv2.RETR_TREE tells OpenCV to compute the hierarchy 
# (relationship) between contours,
# We could have also used the cv2.RETR_LIST option as well;
# To compress the contours to save space using cv2.CV_CHAIN_APPROX_SIMPLE.
img2, cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.cv2.CV_CHAIN_APPROX_NONE)[1]

# Return only the 10 largest contours
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:15]

# Initialize empty list
lst_intensities = []

img = blur.copy()

# For each list of contour points...
# for i in range(len(cnts)):
        # Create a mask img that contains the contour filled in
        # cimg = np.zeros_like(img)
        # cv2.drawContours(cimg, cnts, i, color=255, thickness=-1)

        # Access the img pixels and create a 1D numpy array then add to list
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
                
                # Cortar a ára na imgm original
                x, y, w, h = cv2.boundingRect(approx)
                # make the box a little bigger
                x, y, w, h = x - 8, y - 8, w + 8, h + 8

                # rect = cv2.boundingRect(approx)
                # Mat subMat = new Mat(mRgba, rect)
                # Mat zeroMat = np.zeros(subMat.size(),subMat.type())
                # zeroMat.copyTo(subMat)

                # cv2.imshow("crop", crop)
                # cv2.waitKey(0)

                bouding = img.copy()
                
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

                # TODO: Clean img using mask of the perimeter

                # cv2.imshow("crop", crop)
                # cv2.waitKey(0)

                # Redimencionar imgm
                # resized = _img_resize(crop, width = 180, inter = cv2.INTER_CUBIC)
                # resized = _img_resize(blur, imgRadius)
                resized = imutils.resize(crop, width=180)

                # resized = add_border(resized, output_img='resized.jpg', border=100, color='white')
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
                cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 5)

                cv2.imshow("object", img)
                # cv2.imwrite("object.jpg", img)
                cv2.waitKey(0)

                # res = np.hstack((original, edged, img))

                # Extrair a característica:
                # Encontrar o histograma de cada camada
                # Encontrar a área
                # Encontrar a Variância de projeção vertical
		# Compute Zernike moments to characterize the shape of object outline
                moments = zm.describe(thresh)

                # print(moments)

                # Utilizar o dataset anotado de placas e comparar utilizando distância Euclidiana
                # Return 5 first similarities
                results = searcher.search(indexA, moments)[:10]
                for r in results:
                        img_ref = r[1]
                        img_distance = r[0]

                        if img_distance <= 0.109:
                                print("Objeto semelhante: {} - {}".format(img_distance, img_ref))

                break

cv2.destroyAllWindows()