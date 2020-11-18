import cv2 as cv
import numpy as np
from HysteresisThreshold import HysteresisThreshold
from Non_Max_Suppression import Non_Max_Suppression

if __name__ == "__main__":
    image = cv.imread('Rectangles.jpg')
    image_BnW = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_GaussBlur = cv.GaussianBlur(image_BnW, (3, 3), 1.4)
    image_gradientX = cv.Sobel(image_GaussBlur, cv.CV_64F, 1, 0, ksize=3)
    image_gradientY = cv.Sobel(image_GaussBlur, cv.CV_64F, 0, 1, ksize=3)
    image_gradient_magnitude = np.sqrt(
        (image_gradientX**2) + (image_gradientY**2))
    image_gradient_angle = np.arctan2(image_gradientY, image_gradientX)
    image_after_non_suppression = Non_Max_Suppression(
        image_gradient_magnitude, image_gradient_angle)
    image_after_hysteresis_thresholding = HysteresisThreshold(
        image_after_non_suppression, 0.3, 0.7)

    image_dilated = cv.dilate(image_after_hysteresis_thresholding,
                              cv.getStructuringElement(cv.MORPH_RECT, (3, 3)), iterations=2)

    image_closing = cv.erode(image_dilated, cv.getStructuringElement(
        cv.MORPH_RECT, (3, 3)), iterations=1)

    image_dilated = image_dilated.astype(np.uint8)
    # image_eroded = image_eroded.astype(np.uint8)

    contours, hierarchy = cv.findContours(
        image_dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    print(f"The total number of objects in the given image is {len(contours)}")
    # image_dilated = image_dilated.astype(np.uint64)

    cv.imshow("image", image)
    cv.imshow("image_BnW", image_BnW)
    cv.imshow("image_GaussBlur", image_GaussBlur)
    cv.imshow("image_gradientX", image_gradientX)
    cv.imshow("image_gradientY", image_gradientY)
    cv.imshow("image_gradient_magnitude", image_gradient_magnitude)
    cv.imshow("image_gradient_angle", image_gradient_angle)
    cv.imshow("image_after_non_suppression", image_after_non_suppression)
    cv.imshow("image_after_hysteresis_thresholding",
              image_after_hysteresis_thresholding)
    cv.imshow("image_dilated", image_dilated)
    cv.imshow("image_closing", image_closing)

    cv.waitKey(0)
    cv.destroyAllWindows()
