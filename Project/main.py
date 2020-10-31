import numpy as np
import matplotlib.pyplot as plt
from GaussianBlur import GaussianBlur
from GradientX import GradientX
from GradientY import GradientY
import math as m
import cv2 as cv
import PIL

Laplacian_kernal = np.array([[0.093124, 0.0118914, 0.093124],
                             [0.118914, 0.151845, -0.118914],
                             [0.093124, 0.118914, 0.093124]])

Laplacian_kernal2 = np.array([[0.012841, 0.026743, 0.03415, 0.026743, 0.012841],
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

Blurred_Image = GaussianBlur(
    'lena.jpg', Laplacian_kernal.shape, Laplacian_kernal)

Differentiated_in_X_Direction_Image = GradientX(
    Blurred_Image, Sobel_kernelX.shape, Sobel_kernelX)
print(Differentiated_in_X_Direction_Image)

Differentiated_in_Y_Direction_Image = GradientY(
    Blurred_Image, Sobel_kernelY.shape, Sobel_kernelY)
print(Differentiated_in_Y_Direction_Image)

Sx = cv.imread(Differentiated_in_X_Direction_Image, cv.COLOR_BGR2GRAY)
Sy = cv.imread(Differentiated_in_Y_Direction_Image, cv.COLOR_BGR2GRAY)
Sx = np.array(Sx)
Sy = np.array(Sy)
print(Sx.shape)
# cv.imshow("Check1", Sx)
# cv.imshow("Check2", Sy)
# cv.waitKey(0)


S = np.abs(np.sqrt(Sx ** 2 + Sy ** 2))
print(type(S))
# img = PIL.Image.fromarray(S)
# img.show()


plt.imsave(fname='Magnitude_image.jpg',
           cmap='gray', arr=S, format='jpg')

Theta = (np.arctan2(Sx, Sy)) * (180 / np.pi)
print(Theta)
plt.imsave(fname='Image_edge_angle.jpg',
           cmap='gray', arr=Theta, format='jpg')

# cv.imwrite("Magnitude.jpg", S)
# plt.show()
cv.waitKey(0)
