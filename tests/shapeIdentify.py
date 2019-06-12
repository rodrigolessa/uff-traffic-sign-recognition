import cv2
import numpy as np
from normalization import Normalizer
import imutils

def detectCircle(im):
    # detect circles in the image
    n = Normalizer(170)

    im = n.crop(im)
    new = imutils.resize(im, height=170)
    if new.shape[1] > 170:
        new = imutils.resize(im, width=170)

    circles = cv2.HoughCircles(new, cv2.HOUGH_GRADIENT, 1.5, minDist=170, param2=30, minRadius=70, maxRadius=85)

    return not circles is None

def detectShape(c, img):
    shape = 'unknown'
    # calculate perimeter using
    peri = cv2.arcLength(c, True)
    # apply contour approximation and store the result in vertices
    vertices = cv2.approxPolyDP(c, 0.04 * peri, True)

    # If the shape it triangle, it will have 3 vertices
    if len(vertices) == 3:
        shape = 'triangle'

    # if the shape has 4 vertices, it is either a square or
    # a rectangle
    elif len(vertices) == 4:
        # using the boundingRect method calculate the width and height
        # of enclosing rectange and then calculte aspect ratio

        x, y, width, height = cv2.boundingRect(vertices)
        aspectRatio = float(width) / height

        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        if aspectRatio >= 0.95 and aspectRatio <= 1.05:
            shape = "square"
        else:
            shape = "rectangle"

    # if the shape is a pentagon, it will have 5 vertices
    elif len(vertices) == 5:
        shape = "pentagon"

    # otherwise, we assume the shape is a circle
    elif detectCircle(img):
        shape = "circle"

    # return the name of the shape
    return shape