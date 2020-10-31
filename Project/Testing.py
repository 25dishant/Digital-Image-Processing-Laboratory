import numpy as np
import cv2 as cv
import math


def GaussianBlur(ImageName, kernal_size, Laplacian_kernal):
    # Conversion of Image into a matrix
    image = cv.imread(ImageName)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    print(image.shape)
    ImageMatrix = []
    for r in range(0, image.shape[0]):
        row = []
        for c in range(0, image.shape[1]):
            row.append(image.item(r, c))
        ImageMatrix.append(row)
    ImageMatrix = np.array(ImageMatrix)

    width = len(ImageMatrix[0])
    height = len(ImageMatrix)
    if kernal_size[0] == kernal_size[1] and kernal_size[0] > 2:
        ImageMatrix = np.pad(ImageMatrix, kernal_size[0]-2, mode='constant')
    else:
        pass

    GiantMatrix = []
    for i in range(0, height-kernal_size[1]+1):
        for j in range(0, width-kernal_size[0]+1):
            GiantMatrix.append(
                [
                    [ImageMatrix[col][row]
                        for row in range(j, j + kernal_size[0])]
                    for col in range(i, i + kernal_size[1])
                ])

    Matrix_Sampling = np.array(GiantMatrix)
    print(Matrix_Sampling.shape)

    Transformed_Matrix = []
    Matrix_Sampling = np.array(Matrix_Sampling)
    for each_mat in Matrix_Sampling:
        Transformed_Matrix.append(
            np.sum(np.multiply(each_mat, Laplacian_kernal)))
    reshape_val = int(math.sqrt(Matrix_Sampling.shape[0]))
    Transformed_Matrix = np.array(
        Transformed_Matrix).reshape(reshape_val, reshape_val)

    # print(transform_mat)
    OutputImageName, Extension = ImageName.split('.')
    cv.imwrite(str(OutputImageName+"_GaussianBlurred." +
                   Extension), Transformed_Matrix)


# GaussianBlur('lena.jpg', Laplacian_kernal2.shape, Laplacian_kernal2)
