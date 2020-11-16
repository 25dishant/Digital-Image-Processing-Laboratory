import numpy as np
import cv2 as cv

image = cv.imread("Magnitude_Image.jpg")

# image = cv.dilate(image, cv.getStructuringElement(
#     cv.MORPH_RECT, (3, 3)), iterations=1)
image = cv.erode(image, cv.getStructuringElement(
    cv.MORPH_RECT, (3, 3)), iterations=1)

cv.imshow("Image", image)
cv.waitKey(0)
cv.destroyAllWindows()
