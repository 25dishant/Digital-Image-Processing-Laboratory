import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt

Gaussian_kernal1 = np.array([[0.093124, 0.0118914, 0.093124],
                             [0.118914, 0.151845, 0.118914],
                             [0.093124, 0.118914, 0.093124]])


Gaussian_kernal2 = np.array([[0.102059, 0.115349, 0.102059],
                             [0.115349, 0.130371, 0.115349],
                             [0.102059, 0.115349, 0.102059]])

Gaussian_kernal3 = np.array([[0.012841, 0.026743, 0.03415, 0.026743, 0.012841],
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


def GaussianBlur(ImageName, kernel_size, Gaussian_kernal):
    # Conversion of Image into a matrix
    image = cv.imread(ImageName)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # Open Image Grayscale Mode

    ImageMatrix = []  # Initialise a list to keep the Image in matrix form
    for r in range(0, image.shape[0]):
        row = []
        for c in range(0, image.shape[1]):
            row.append(image.item(r, c))
        ImageMatrix.append(row)
    # We have image in the form of matrix at this point.
    ImageMatrix = np.array(ImageMatrix)

    width = len(ImageMatrix[0])  # Width of the Image Matrix
    height = len(ImageMatrix)  # Height of the Image Matrix

    # Condition to check the squared kernel
    if kernel_size[0] == kernel_size[1] and kernel_size[0] > 2:
        # Pad the image to avoid any loss of information after convolution
        ImageMatrix = np.pad(ImageMatrix, kernel_size[0]-2, mode='constant')
    else:
        pass

    Main_Matrix = []
    for i in range(0, height-kernel_size[1]+1):
        for j in range(0, width-kernel_size[0]+1):
            Main_Matrix.append([
                [ImageMatrix[col][row]
                 for row in range(j, j + kernel_size[0])]
                for col in range(i, i + kernel_size[1])
            ])

    Main_Matrix = np.array(Main_Matrix)

    Transformed_Matrix = []
    Main_Matrix = np.array(Main_Matrix)
    for Submatrix in Main_Matrix:
        Transformed_Matrix.append(
            np.sum(np.multiply(Submatrix, Gaussian_kernal)))
    reshape_val = int(math.sqrt(Main_Matrix.shape[0]))
    Transformed_Matrix = np.array(
        Transformed_Matrix).reshape(reshape_val, reshape_val)

    # Convert the Tranformed Matrix into an image and save it with a proper name.
    Name, Extension = ImageName.split('.')
    OutputImageName = str(Name+"_GaussianBlurred."+Extension)
    cv.imwrite(OutputImageName, Transformed_Matrix)
    return OutputImageName


# Function for the gradient
# It takes the Gaussian blurred image and kernel as the input and returns the gradient of the image
def Gradient(image, kernel):
    image = cv.imread(image)

    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros(
        (image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height,
                 pad_width:padded_image.shape[1] - pad_width] = image

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(
                kernel * padded_image[row:row + kernel_row, col:col + kernel_col])

    return output

# This is the non maximaum suppression fucntion without interpolation.


def Non_Max_Suppression(Image, Angle):
    # Get the Image dimensions
    Image_Height = Image.shape[0]
    Image_Width = Image.shape[1]

    # Create a zero matrix with dimensions of image size
    Image_Matrix = np.zeros((Image_Height, Image_Width))
    for h in range(2, Image_Height-1):
        for w in range(2, Image_Width-1):
            if (-22.5 <= Angle[h, w] <= 22.5) or (-157.5 > Angle[h, w] >= -180):
                if Image[h, w] >= Image[h, w+1] and Image[h, w] >= Image[h, w-1]:
                    Image_Matrix[h, w] = Image[h, w]
                else:
                    Image_Matrix[h, w] = 0

            elif (22.5 <= Angle[h, w] <= 67.5) or (-112.5 > Angle[h, w] >= -157.5):
                if Image[h, w] >= Image[h+1, w+1] and Image[h, w] >= Image[h-1, w-1]:
                    Image_Matrix[h, w] = Image[h, w]
                else:
                    Image_Matrix[h, w] = 0

            elif (67.5 <= Angle[h, w] <= 112.5) or (-67.5 > Angle[h, w] >= -112.5):
                if Image[h, w] >= Image[h+1, w] and Image[h, w] >= Image[h-1, w]:
                    Image_Matrix[h, w] = Image[h, w]
                else:
                    Image_Matrix[h, w] = 0

            elif (112.5 <= Angle[h, w] <= 157.5) or (-22.5 > Angle[h, w] >= -67.5):
                if Image[h, w] >= Image[h+1, w-1] and Image[h, w] >= Image[h-1, w+1]:
                    Image_Matrix[h, w] = Image[h, w]
                else:
                    Image_Matrix[h, w] = 0

    return Image_Matrix

# A function for the double thresholding of the image


def HysteresisThreshold(Image, low, high):
    Image_Height = Image.shape[0]
    Image_Width = Image.shape[1]

    High_Threshold = np.max(Image)*high
    Low_Threshold = High_Threshold*low

    for h in range(2, Image_Height-1):
        for w in range(2, Image_Width-1):
            if Image[h, w] > High_Threshold:
                Image[h, w] = 1
            elif Image[h, w] < Low_Threshold:
                Image[h, w] = 0
            else:
                Image[h, w] = 0.5

    return Image


if __name__ == "__main__":

    ImageName = input(
        "Enter the name of Image alongwith Extension \n (Image must be in the same folder)\n")

    BlurredImage = GaussianBlur(
        ImageName, Gaussian_kernal1.shape, Gaussian_kernal1)

    ImageX = Gradient(BlurredImage, Sobel_kernelX)

    plt.imsave(fname='ImageX.jpg',
               cmap='gray', arr=ImageX, format='jpg')

    ImageY = Gradient(BlurredImage, Sobel_kernelY)

    plt.imsave(fname='ImageY.jpg',
               cmap='gray', arr=ImageY, format='jpg')

    Magnitude_Image = np.sqrt(ImageX**2 + ImageY**2)

    plt.imsave(fname='Magnitude_Image.jpg',
               cmap='gray', arr=Magnitude_Image, format='jpg')

    Angle_Image = np.arctan2(ImageY, ImageX)

    plt.imsave(fname='Angle_Image.jpg',
               cmap='gray', arr=Angle_Image, format='jpg')

    Image_NonMaxSuppression = Non_Max_Suppression(Magnitude_Image, Angle_Image)

    plt.imsave(fname='Image_NonMaxSuppression.jpg',
               cmap='gray', arr=Image_NonMaxSuppression, format='jpg')

    Image_HysteresisThreshold = HysteresisThreshold(
        Image_NonMaxSuppression, 0.3, 0.7)

    # cv.imwrite("Image_HysteresisThreshold.png", Image_HysteresisThreshold)

    plt.imsave(fname='EdgeImage.jpg',
               cmap='gray', arr=Image_HysteresisThreshold, format='jpg')

    cv.imshow("ImageX", ImageX)
    cv.imshow("ImageY", ImageY)
    cv.imshow("Magnitude_Image", Magnitude_Image)
    cv.imshow("Angle_Image", Angle_Image)
    cv.imshow("Image_NonMaxSuppression", Image_NonMaxSuppression)
    cv.imshow("Image_HysteresisThreshold", Image_HysteresisThreshold)
    cv.waitKey(0)
    cv.destroyAllWindows()
