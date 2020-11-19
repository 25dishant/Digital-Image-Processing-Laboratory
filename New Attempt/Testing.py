import numpy as np
import cv2 as cv

image = cv.imread("EdgeImage.jpg")


image_BnW = cv.cvtColor(image, cv.COLOR_BGR2GRAY)


image_dilate = cv.dilate(
    image_BnW,
    cv.getStructuringElement(cv.MORPH_RECT, (3, 3)),
    iterations=1)


image_erode = cv.erode(
    image_dilate,
    cv.getStructuringElement(cv.MORPH_RECT, (3, 3)),
    iterations=1)


lines = cv.HoughLinesP(image_erode, 1, np.pi/180, 30)

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv.line(image, (x1, y1), (x2, y2), (255, 255, 255), 1)

coutours, hierarchy = cv.findContours(
    image_, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
print(f"Number of objects is {len(coutours)}")


cv.imshow("image", image)
cv.imshow("image_BnW", image_BnW)
cv.imshow("image_dilate", image_dilate)
cv.imshow("image_erode", image_erode)

cv.waitKey(0)
cv.destroyAllWindows()
