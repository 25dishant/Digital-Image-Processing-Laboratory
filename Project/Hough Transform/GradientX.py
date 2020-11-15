import numpy as np
import cv2 as cv
import math


def GradientX(ImageName, kernel_size, Sobel_kernelX):
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

    GiantMatrix = []
    for i in range(0, height-kernel_size[1]+1):
        for j in range(0, width-kernel_size[0]+1):
            GiantMatrix.append([
                [ImageMatrix[col][row]
                 for row in range(j, j + kernel_size[0])]
                for col in range(i, i + kernel_size[1])
            ])

    Matrix_Sampling = np.array(GiantMatrix)

    Transformed_Matrix = []
    Matrix_Sampling = np.array(Matrix_Sampling)
    for each_mat in Matrix_Sampling:
        Transformed_Matrix.append(
            np.sum(np.multiply(each_mat, Sobel_kernelX)))
    reshape_val = int(math.sqrt(Matrix_Sampling.shape[0]))
    Transformed_Matrix = np.array(
        Transformed_Matrix).reshape(reshape_val, reshape_val)

    # Convert the Tranformed Matrix into an image and save it with a proper name.
    Name, Extension = ImageName.split('.')
    OutputImageName = str(Name+"_GradientX."+Extension)
    cv.imwrite(OutputImageName, Transformed_Matrix)
    return OutputImageName
