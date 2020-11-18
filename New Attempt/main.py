import cv2 as cv
import numpy as np
from GaussianBlur import GaussianBlur
from Gradient import Gradient
from Non_Max_Suppression import Non_Max_Suppression
from HysteresisThreshold import HysteresisThreshold


Laplacian_kernal1 = np.array([[0.093124, 0.0118914, 0.093124],
                              [0.118914, 0.151845, 0.118914],
                              [0.093124, 0.118914, 0.093124]])


Laplacian_kernal2 = np.array([[0.102059, 0.115349, 0.102059],
                              [0.115349, 0.130371, 0.115349],
                              [0.102059, 0.115349, 0.102059]])

Laplacian_kernal3 = np.array([[0.012841, 0.026743, 0.03415, 0.026743, 0.012841],
                              [0.026743, 0.055697, 0.071122, 0.055697, 0.026743],
                              [0.03415, 0.071122, 0.090818, 0.071122, 0.03415],
                              [0.026743, 0.055697, 0.071122, 0.055697, 0.026743],
                              [0.012841, 0.026743, 0.03415, 0.026743, 0.012841]])

Sobel_kernelX = np.array([[1, 0, -1],
                          [2, 0, -2],
                          [1, 0, -1]])

Sobel_kernelY = np.array([[1, 2, 1],
                          [0, 0, 0],
                          [-1, -2, -1]])


Sobel_kernelX2 = np.array([[-1, -2, 0, 2, 1],
                           [-4, -10, 0, 10, 4],
                           [-7, -17, 0, 17, 7],
                           [-4, -10, 0, 10, 4],
                           [-1, -2, 0, -2, 1]])

Sobel_kernelY2 = np.array([[1, 4, 7, 4, 1],
                           [2, 10, 17, 10, 2],
                           [0, 0, 0, 0, 0],
                           [-2, -10, -17, -10, -2],
                           [-1, -4, -7, -4, -1]])


if __name__ == "__main__":
    BlurredImage = GaussianBlur(
        "Rectangles.jpg", Laplacian_kernal1.shape, Laplacian_kernal1)

    ImageX = Gradient(BlurredImage, Sobel_kernelX)

    ImageY = Gradient(BlurredImage, Sobel_kernelY)

    Magnitude_Image = np.sqrt(ImageX**2 + ImageY**2)

    Angle_Image = np.arctan2(ImageY, ImageX)

    Image_NonMaxSuppression = Non_Max_Suppression(Magnitude_Image, Angle_Image)

    Image_HysteresisThreshold = HysteresisThreshold(
        Image_NonMaxSuppression, 0.3, 0.7)

    cv.imshow("ImageX", ImageX)
    cv.imshow("ImageY", ImageY)
    cv.imshow("Magnitude_Image", Magnitude_Image)
    cv.imshow("Angle_Image", Angle_Image)
    cv.imshow("Image_NonMaxSuppression", Image_NonMaxSuppression)
    cv.imshow("Image_HysImage_HysteresisThreshold", Image_HysteresisThreshold)
    cv.waitKey(0)
    cv.destroyAllWindows()
