# The main goal is to find the screen of Game and highlight it, 

# iIport the necessary packages
# The image_utils contains convenience methods to handle basic image processing techniques
# resizing, rotating, and translating. 
import imutils
import numpy as np
#import argparse
import cv2
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
#imageName = "dataset_traffic_sign/400px-Give_way_outdoor.jpg"
#imageName = "dataset_traffic_sign/41-kDNLiI4L._SX425_.jpg"
#imageName = "dataset_traffic_sign/141441.jpg"
imageName = "dataset_traffic_sign/RW_000005-2.jpg"
imageNumber = "4"
imageRef = "traffic_sign_canny_douglaspeucker_"

# Load the query image, 
image = cv2.imread(imageName)

# Only for the test image we have
# image = imutils.rotate(image, 90, center = None, scale = 1.0)

# compute the ratio of the old height
# to the new height, clone it, and resize it
ratio = image.shape[0] / 200.0

# resize it - The smaller the image is, the faster it is to process
image = imutils.resize(image, height = 300)

# clone it
#original = image.copy()
cv2.imwrite('traffic_sign_canny_douglaspeucker_' + imageNumber + '1.jpg', image)

print(imageRef + imageNumber + "3.jpg")

# Convert the image to grayscale, blur it, 
# and find edges in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Blur the image slightly by using the cv2.bilateralFilter function
# Bilateral filtering has the nice property of removing noise in the image 
# while still preserving the actual edges.
gray = cv2.bilateralFilter(gray, 11, 17, 17)

# Canny edge detector finds edge like regions in the image
edged = cv2.Canny(gray, 30, 200)

# Debugging:
cv2.imshow('Canny', edged)
cv2.waitKey(0)

cv2.imwrite('traffic_sign_canny_douglaspeucker_' + imageNumber + '2.jpg', edged)

# Find contours in the edged image, keep only the largest ones, 
# and initialize our screen contour:
# The cv2.findContours function gives us a list of contours that it has found.
# The second parameter cv2.RETR_TREE tells OpenCV to compute the hierarchy 
# (relationship) between contours,
# We could have also used the cv2.RETR_LIST option as well;
# To compress the contours to save space using cv2.CV_CHAIN_APPROX_SIMPLE.
image2, cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Return only the 10 largest contours
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]

# Initialize screenCnt, the contour that corresponds to our object to find
screenCnt = None

# Loop over contours
for c in cnts:

	#print(c)

    # cv2.arcLength and cv2.approxPolyDP. 
    # These methods are used to approximate the polygonal curves of a contour.
	peri = cv2.arcLength(c, True)

	print(peri)

    # Level of approximation precision. 
    # In this case, we use 2% of the perimeter of the contour.
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	print(approx)

    # we know that a Object screen is a rectangle,
    # and we know that a rectangle has four sides, thus has four vertices.
	# If our approximated contour has four points, then
	# we can assume that we have found our screen.
	if len(approx) == 3:
		screenCnt = approx
		break

# Drawing our screen contours, we can clearly see that we have found the Object screen
#if isinstance(screenCnt, list):
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 5)
cv2.imshow("Object Screen", image)
cv2.waitKey(0)

cv2.imwrite('traffic_sign_canny_douglaspeucker_' + imageNumber + '3.jpg', image)

#res = np.hstack((original, edged, image))

#cv2.imwrite('traffic_sign_canny_douglaspeucker_1.jpg', res)
#cv2.imshow("Resultados", res)
#cv2.waitKey(0)

cv2.destroyAllWindows()