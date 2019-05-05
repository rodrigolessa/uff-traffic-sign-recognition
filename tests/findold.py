# Indexing the dataset by quantifying each image in terms of shape.
# Apply the shape descriptor defined to every sprite in dataset.
# Frist we need the outline (or mask) of the object in the image 
# prior to applying Zernike moments. 
# In order to find the outline, we need to apply segmentation

# Import the necessary packages
from zernike_moments import ZernikeMoments
from searcher import Searcher
import numpy as np
import cv2
import pickle as cp
import glob
import os
import sys

class MainFind:

    def __init__(self):

        # Load the index
        # Initializing
        self.imageMomentsFile = 'index.pkl'
        #self.index = open(self.imageMomentsFile, 'rb')
        #self.index = cp.load(self.index)
        # Initialize descriptor with a radius of 21 pixels
        # radius : integer
        # the maximum radius for the Zernike polynomials, in pixels. 
        # Note that the area outside the circle (centered on center of mass) 
        # defined by this radius is ignored.
        #self.zm = ZernikeMoments(21)

        # If index file exists, try to delete
        #try:
        #os.remove(self.imageMomentsFile)
        #except OSError:
        #pass

    def ExtractShape(self, imgPath):
        # Extract image name, this will serve as unqiue key into the index dictionary
        imgName = ''#imgPath[spritePath.rfind('\\') + 1:].replace(imageExtension, '')
        
        # then load the image.
        img = cv2.imread(imgPath)

        # Pad the image with extra white pixels to ensure the
        # edges of the object are not up against the borders
        # of the image
        #img = cv2.copyMakeBorder(img, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value = 255)
        
        img = cv2.resize(img, None, fx=0.75, fy=0.75, interpolation = cv2.INTER_CUBIC)

        # Debugging: Show Original
        cv2.imshow('original', img)
        cv2.waitKey(0)

        bordersize = 100

        #bordered = cv2.copyMakeBorder(img, 
        #    top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, 
        #    borderType= cv2.BORDER_CONSTANT, 
        #    value=[255,255,255])

        ######################################################
        # Espaços de cores da imagem

        ######################################################
        # Canais da Imagem

        # Convert it to grayscale
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Debugging: Show converted,one chanel with  intensity
        cv2.imshow('gray', grayscale)
        cv2.imwrite("grayscale.jpg", grayscale) # save frame as JPEG file
        cv2.waitKey(0)

        ######################################################
        # Verificar se a imagem possui baixo constraste

        # The low contrast fraction threshold. An image is considered low-contrast 
        # when its range of brightness spans less than this fraction of its data type’s 
        # full range. [1]

        #skimage.exposure.is_low_contrast(image, fraction_threshold=0.05, lower_percentile=1, upper_percentile=99, method='linear')

        ######################################################
        # Equalização baseado em histograma da imagem

        #CLAHE (Contrast Limited Adaptive Histogram Equalization)

        # create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        #clahe = cv2.createCLAHE()

        cl1 = clahe.apply(grayscale)

        #res = np.hstack((img, cl1))
        #cv2.imwrite('res.png', res)
        cv2.imshow("Equalization", cl1)
        cv2.imwrite("eualization.jpg", cl1) # save frame as JPEG file
        cv2.waitKey(0)

        ######################################################
        # Algoritmos de redução de ruido:

        # Apply a blur filter to reduce noise
        #blur = cv2.medianBlur(cl1, 5)

        #suave = cv2.GaussianBlur(grayscale, (11, 11), 0)

        # Bilateral Filter can reduce unwanted noise
        blur = cv2.bilateralFilter(cl1, 9, 75, 75)
        #blur = cv2.bilateralFilter(cl1, 7, 49, 49)
        
        # "fastNlMeansDenoising"
        
        #row, col = cl1.shape
        #mean = 0

        #gauss = np.random.normal(mean, 1, (row, col))
        #gauss = gauss.reshape(row, col)
        
        #noisy = cl1 + gauss
        #noisy = (noisy).astype('uint8')

        #cv2.imshow('Blur', noisy)
        #cv2.waitKey(0)

        #for i in range(3):

            #noisy = cv2.fastNlMeansDenoising(noisy, templateWindowSize=5, searchWindowSize=25, h=65)
            #cv2.imshow('Blur', noisy)
            #cv2.waitKey(0)
            #noisy = cv2.threshold(noisy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            #_, noisy = cv2.threshold(noisy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # self.view(img, noisy)
        #cv2.imshow('Blur', noisy)

        #cv2.imwrite(os.path.join(debugPath , '{}_blur.png'.format(imageName)), blur)

        # Apply a blur filter to reduce noise
        blur = cv2.medianBlur(blur, 5)

        cv2.imshow('Blur', blur)
        cv2.imwrite("blur.jpg", blur) # save frame as JPEG file
        cv2.waitKey(0)

        ######################################################
        # Inverter intensidades

        # For segmentation: Flip the values of the pixels 
        # (black pixels are turned to white, and white pixels to black).
        #inv = cv2.bitwise_not(blur)

        # Debugging: Invert image
        #cv2.imshow('Inverted', inv)
        #cv2.waitKey(0)

        ######################################################
        # Limiar na imagem

        # Then, any pixel with a value greater than zero (black) is set to 255 (white)
        #inv[inv > 0] = 255
        #thresh = inv

        # Canny edge detector finds edge like regions in the image
        #edged = cv2.Canny(blur, 30, 200)

        # Debugging:
        #cv2.imshow("Canny", edged)
        #cv2.waitKey(0)


        # Then, any pixel with a value greater than zero (black) is set to 255 (white)
        # NOTE: First version
        #thresh[thresh > 0] = 255
        # NOTE: Second version
        # THRESH_BINARY = fundo preto or THRESH_BINARY_INV = fundo branco
        # Then, any pixel with a value greater than zero (black) is set to 255 (white)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Debugging: Threshold it
        cv2.imshow('thresholded', thresh)
        cv2.imwrite("thresholded.jpg", thresh) # save frame as JPEG file
        cv2.waitKey(0)

        ######################################################
        # Reescrevendo os contornos nas imagens

        # First, we need a blank image to store outlines
        # we appropriately a variable called outline 
        # and fill it with zeros with the same width and height as the sprite image.

        # Accessing Image Properties
        # Image properties include number of rows, columns and channels, 
        # type of image data, number of pixels etc.
        # Shape of image is accessed by img.shape. It returns a tuple of number of rows, 
        # columns and channels (if image is color):
        outline = np.zeros(img.shape, dtype = "uint8")

        ######################################################
        # Detectar contornos na imagem

        # Initialize the outline image,
        # find the outermost contours (the outline) of the object, 
        # cv2.RETR_EXTERNAL - telling OpenCV to find only the outermost contours.
        # cv2.CHAIN_APPROX_SIMPLE - to compress and approximate the contours to save memory
        img2, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort the contours based on their area, in descending order. 
        # keep only the largest contour and discard the others.
        #contours = sorted(contours, key = cv2.contourArea, reverse = True)[0]
        contours = sorted(contours, key = cv2.contourArea, reverse = True)[3]

        cv2.drawContours(outline, [contours], -1, 255, -1)
        cv2.imshow('outline', outline)
        #cv2.imwrite(imageRef + imageNumber + "3.jpg", image)
        cv2.waitKey(0)

        print('Total of contours')
        print(len(contours))

        # Load the index
        index = open('index.pkl', 'rb')
        index = cp.load(index)

        desc = ZernikeMoments(8)

        queryFeatures = desc.describe(outline)
                    
        # Perform the search to identify the pokemon
        searcher = Searcher(index)
        # Return 10 first similarities
        results = searcher.search(queryFeatures)[:5]

        print("That object is: {}".format(results[0][1].upper()))

        # Loop over contours
        for c in contours:
            # cv2.arcLength and cv2.approxPolyDP. 
            # These methods are used to approximate the polygonal curves of a contour.
            peri = cv2.arcLength(c, True)

            if (peri > 600):

                print(peri)
                #641.4041084051132

                area = cv2.contourArea(c)

                if (area > 600):

                    21739

                    print(area)
                    #2219.5

                    #outlineIsolated = np.zeros(img.shape, dtype = "uint8")

                    # The outline is drawn as a filled in mask with white pixels:
                    #cv2.drawContours(outlineIsolated, [c], -1, 255, -1)
                    #cv2.imshow('outline', outlineIsolated)
                    #cv2.waitKey(0)

                    #queryFeatures = self.zm.describe(outlineIsolated)
                    
                    # Perform the search to identify the pokemon
                    #searcher = Searcher(self.index)
                    # Return ? first similarities
                    #results = searcher.search(queryFeatures)[:5]

                    #print(results)

                    #print("Object: {}".format(results[0][1].upper()))

                    # The outline is drawn as a filled in mask with white pixels:
                    #cv2.drawContours(outline, [c], -1, 255, -1)
                    #cv2.imshow('outline', outline)
                    #cv2.imwrite(imageRef + imageNumber + "3.jpg", image)
                    #cv2.waitKey(0)
        
        # The outline is drawn as a filled in mask with white pixels:
        #cv2.drawContours(outline, contours[:10], -1, 255, -1)

        # Drawing our screen contours, we can clearly see that we have found the Object screen
        #cv2.drawContours(outline, [contours], -1, (0, 255, 0), 3)

        # Debugging: just outline of the object
        #cv2.imshow('outline', outline)
        #cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Compute Zernike moments to characterize the shape of object outline
        #moments = self.zm.describe(outline)

        # then update the index
        #self.index[imageName] = moments

        # cPickle for writing the index in a file
        #with open(imageMomentsFile, "wb") as outputFile:
	        #cp.dump(index, outputFile, protocol=cp.HIGHEST_PROTOCOL)

mi = MainFind()
mi.ExtractShape('frame114.jpg')