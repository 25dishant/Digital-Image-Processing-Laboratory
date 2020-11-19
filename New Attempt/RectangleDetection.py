import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

image = cv.imread("Rectangles.jpg")

grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# edgeImage = cv.Canny(grayImage, 75, 150)

# plt.imsave(fname='EdgeImage.jpg',
#            cmap='gray', arr=edgeImage, format='jpg')

edgeImage = cv.imread("EdgeImage.jpg")

edgeImage = cv.cvtColor(edgeImage, cv.COLOR_BGR2GRAY)

# image_dilate = cv.dilate(
#     edgeImage,
#     cv.getStructuringElement(cv.MORPH_RECT, (5, 5)),
#     iterations=1)

# image_erode = cv.erode(
#     image_dilate,
#     cv.getStructuringElement(cv.MORPH_RECT, (3, 3)),
#     iterations=1)

lines = cv.HoughLinesP(edgeImage, 1, np.pi/180, 50)

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)


coutours, hierarchy = cv.findContours(
    edgeImage, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
print(f"Number of objects is {math.ceil(len(coutours)/4)}")

cv.imshow("Image", image)
cv.imshow("EdgeImage", edgeImage)
# cv.imshow("image_dilate", image_dilate)
# cv.imshow("image_erode", image_erode)

cv.waitKey(0)
cv.destroyAllWindows()
