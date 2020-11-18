import numpy as np
import cv2 as cv

image1 = cv.imread('lena.jpg')
image2 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
image3 = cv.GaussianBlur(image2, (3, 3), 1.4)
image4 = cv.Sobel(image3, , 3, 3)

cv.imshow('Image1', image1)
cv.imshow('Image2', image2)
cv.imshow('Image3', image3)
cv.imshow('Image4', image4)
cv.waitKey(0)
cv.destroyAllWindows()
