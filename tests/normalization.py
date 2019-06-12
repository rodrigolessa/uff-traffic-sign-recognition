import cv2
import imutils
import numpy as np
import os
import shutil

class Normalizer():

    def __init__(self, size):
        self.size = size


    def crop(self, img):
        y, x = img.shape
        # bottom= img[y-2:y, 0:x]
        # mean= cv2.mean(bottom)[0]

        # bordersize=100
        # img = cv2.copyMakeBorder(img, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[255,255,255] )

        blur = cv2.bilateralFilter(img, 9, 75, 75)

        # threshold to get just the signature
        retval, thresh_gray = cv2.threshold(blur, thresh=200, maxval=255, type=cv2.THRESH_BINARY)

        # find where the signature is and make a cropped region
        points = np.argwhere(thresh_gray==0) # find where the black pixels are
        points = np.fliplr(points) # store them in x,y coordinates instead of row,col indices
        x, y, w, h = cv2.boundingRect(points) # create a rectangle around those points
        del points
        x, y, w, h = x-10, y-10, w+20, h+20 # make the box a little bigger
        if x < 0: x = 0
        if y < 0: y =0

        return img[y:y+h, x:x+w]


    def scale(self, image):
        new = imutils.resize(image, height=self.size)
        if new.shape[1] > self.size:
            new = imutils.resize(image, width=self.size)

        border_size_x = (self.size - new.shape[1])//2
        border_size_y = (self.size - new.shape[0])//2

        new = cv2.copyMakeBorder(new, border_size_y+self.size, border_size_y+self.size, border_size_x + self.size, border_size_x + self.size, cv2.BORDER_REPLICATE)

        return new


    def removeNoise(self, img):

        bordersize=100
        image = cv2.copyMakeBorder(img, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[255,255,255] )
        
        row,col= image.shape
        mean = 0
        gauss = np.random.normal(mean,1,(row,col))
        gauss = gauss.reshape(row,col)
        noisy = image + gauss
        noisy = (noisy).astype('uint8')
        
        for i in range(3):

            noisy = cv2.fastNlMeansDenoising(noisy, templateWindowSize=5, searchWindowSize=25, h=65)

        noisy = cv2.threshold(noisy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # self.view(img, noisy)

        return noisy


    def view(self, original, normalized):
        cv2.imshow('Original', original)
        cv2.imshow('Normalized', normalized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
     

    def normalize_im(self, im):
        # im = cv2.imread(image, 0)
        
        # im = cv2.bitwise_not(im)

        # i = im.copy()
        
        im = self.removeNoise(im)  #remove noise
        im = self.crop(im)  # get only important shape from image
        im = self.scale(im)  # resize and add border

        # self.view(i, im)
        
        return im