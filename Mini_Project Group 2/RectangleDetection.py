import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

image = cv.imread("Rectangles.jpg")

grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)


edgeImage = cv.imread("edgeImage.jpg")

edgeImage = cv.cvtColor(edgeImage, cv.COLOR_BGR2GRAY)


lines = cv.HoughLinesP(edgeImage, 1, np.pi/180, 50)

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)


coutours, hierarchy = cv.findContours(
    edgeImage, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
print(
    f"Total number of Rectangles in this image is  {math.ceil(len(coutours)/4)}")

cv.imshow("Image", image)
cv.imshow("EdgeImage", edgeImage)

cv.waitKey(0)
cv.destroyAllWindows()
