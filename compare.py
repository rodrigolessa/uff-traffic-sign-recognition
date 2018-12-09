# Indexing the dataset by quantifying each image in terms of shape.
# Apply the shape descriptor defined to every sprite in dataset.
# Frist we need the outline (or mask) of the object in the image 
# prior to applying Zernike moments. 
# In order to find the outline, we need to apply segmentation

# Import the necessary packages
from zernike_moments import ZernikeMoments
from PIL import Image, ImageOps
import numpy as np
import argparse
import cv2
import pickle as cp
import glob
import imutils
# import myutils as ut
# Managing windows files
import os
import sys

# Construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-f", "--folder", required = True, help = "Path to where the files has stored")
#ap.add_argument("-e", "--extension", required = True, help = "Extension of the images")
#ap.add_argument("-i", "--index", required = True, help = "Path to where the index file will be stored")

#args = vars(ap.parse_args())

#imageFolder = args["folder"] #'logos'
imageFolder = "sinalizacao_brasileira_definicao\\selecionados"
#imageExtension = '.' + args["extension"].lower() #'.png'
imageExtension = '.jpg'
imageFinder = '{}\\*{}'.format(imageFolder, imageExtension)
#imageMomentsFile = args["index"] #'index.pkl'
imageMomentsFile = 'index.pkl'
imageRadius = 180

index = {}

try:
	# If index file exists, try to delete
    os.remove(imageMomentsFile)
	# If folder to hold thresholder exists, try to delete
	os.remove(imageFolder + '\\threshold')
except OSError:
    pass

try:
    os.makedirs(imageFolder + '\\threshold')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

# Initialize descriptor with a radius of 160 pixels
zm = ZernikeMoments(imageRadius)

#print(imageFinder)

imagesInFolder = glob.glob(imageFinder)

qt = len(imagesInFolder)

#print('images in the folder: {}'.format(qt))

i = 1

# Loop over the sprite images
for spritePath in imagesInFolder:

	# Extract image name, this will serve as unqiue key into the index dictionary.
	imageName = spritePath[spritePath.rfind('\\') + 1:].lower().replace(imageExtension, '')

	#print('images name: {}'.format(imageName))

	progress(i, qt)

	# then load the image.
	original = cv2.imread(spritePath)

	# Convert it to grayscale
	grayscale = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

	# Bilateral Filter can reduce unwanted noise very well
	blur = cv2.bilateralFilter(grayscale, 9, 75, 75)

	# Redimencionar imagem
	# resized = _img_resize(crop, width = 180, inter = cv2.INTER_CUBIC)
	# resized = _img_resize(blur, imageRadius)
	resized = imutils.resize(blur, width=180)

	#resized = add_border(resized, output_image='resized.jpg', border=100, color='white')
	row, col = resized.shape[:2]
	bottom = resized[row-2:row, 0:col]
	mean = cv2.mean(bottom)[0]

	bordersize = 100
	bordered = cv2.copyMakeBorder(resized, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[mean,mean,mean] )

	# Then, any pixel with a value greater than zero (black) is set to 255 (white)
	_, thresh = cv2.threshold(bordered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

	thresh = cv2.bitwise_not(thresh)

	# debbuging:
	# cv2.imshow("Threshold", thresh)
	# cv2.waitKey(0)

	# Compute Zernike moments to characterize the shape of object outline
	moments = zm.describe(thresh)

	# then update the index
	index[imageName] = moments

	i+=1	

cv2.destroyAllWindows()

# cPickle for writing the index in a file
with open(imageMomentsFile, "wb") as outputFile:
	cp.dump(index, outputFile, protocol=cp.HIGHEST_PROTOCOL)