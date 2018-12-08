# The main goal is to find the screen of Game and highlight it, 

# iIport the necessary packages
# The image_utils contains convenience methods to handle basic image processing techniques
# resizing, rotating, and translating. 
import imutils
import numpy as np
#import argparse
import cv2
import matplotlib.pyplot as plt
import glob
# Managing windows files
import os
import sys

# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-q", "--query", required = True, help = "Path to the query image")
# Only need one command line argument: 
# --query points to the path to where query image is stored on disk.
#args = vars(ap.parse_args())

imageFolder = 'dataset_traffic_sign_sweden'
imageExtension = '.png'
imageFinder = '{}/*{}'.format(imageFolder, imageExtension)

imagesInFolder = glob.glob(imageFinder)

i = 0

# Loop over the sprite images
for spritePath in imagesInFolder:

        # Extract image name, this will serve as unqiue key into the index dictionary.
        imageName = spritePath[spritePath.rfind('\\') + 1:].lower().replace(imageExtension, '')

        i = i + 1

        if (i > 5):
                break

        # Try to manipulate the image if it is possible
        try:

                # then load the image.
                image = cv2.imread(spritePath)

                # Convert it to grayscale
                grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Bilateral Filter can reduce unwanted noise very well
                blur = cv2.bilateralFilter(grayscale, 9, 75, 75)

                # Then, any pixel with a value greater than zero (black) is set to 255 (white)
                _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                thresh = cv2.bitwise_not(thresh)
                #thresh[thresh > 0] = 255

                cv2.imshow("Crop the original", thresh)
                cv2.waitKey(0)

                img_row_sum = np.sum(thresh, axis=1).tolist()
                plt.plot(img_row_sum)
                plt.show()
                #cv2.waitKey(0)
                # Extrair a característica:
                # Encontrar o histograma de cada camada
                # Binarização e limiar com Otsu
                # Extrair as características:
                # Encontrar a área
                # Encontrar a Variância de projeção vertical
                # Utilizar o dataset anotado de placas e comparar utilizando distância Euclidiana



                # Compute Zernike moments to characterize the shape of object outline
                #moments = zm.describe(thresh)

                # then update the index
                #index[imageName] = moments

        except:
                pass


cv2.destroyAllWindows()