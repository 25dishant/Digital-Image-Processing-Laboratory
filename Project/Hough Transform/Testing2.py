import cv2 as cv
import numpy as np

image = cv.imread('Rectangles.jpg')
image2 = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
image2 = cv.Canny(image2, 75, 150)
image2 = cv.dilate(image2, cv.getStructuringElement(
    cv.MORPH_RECT, (5, 5)), iterations=2)

image2 = cv.erode(image2, cv.getStructuringElement(
    cv.MORPH_RECT, (5, 5)), iterations=2)

lines = cv.HoughLinesP(image2, 1, np.pi/180, 30)

coutours, hierarchy = cv.findContours(
    image2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
print(f"Number of objects is {len(coutours)}")
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv.line(image, (x1, y1), (x2, y2), (255, 255, 255), 1)
# print(lines)
cv.imshow('Check', image)


cv.imshow('Check2', image2)

cv.waitKey(0)


# image = cv.imread("Image_After_Hysteresis_Thresholding.jpg")
# image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# image = cv.dilate(image, cv.getStructuringElement(
#     cv.MORPH_RECT, (5, 5)), iterations=1)
# image = cv.erode(image, cv.getStructuringElement(
#     cv.MORPH_RECT, (3, 3)), iterations=1)

# lines = cv.HoughLinesP(image, 1, np.pi/180, 30, maxLineGap=50)
# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     cv.line(image, (x1, y1), (x2, y2), (255, 255, 255), 1)


# coutours, hierarchy = cv.findContours(
#     image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

# print(f"Number of objects is {len(coutours)}")

# cv.imshow('Image', image)
# cv.waitKey(0)
# cv.destroyAllWindows()
