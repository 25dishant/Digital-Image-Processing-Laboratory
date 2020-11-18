import numpy as np
import cv2 as cv
import math
# import matplotlib.pyplot as plt


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
    if (kernel_size[0] == kernel_size[1]) and (kernel_size[0] > 2):
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


# def GradientX(image, kernel, average=False, verbose=False):
#     image = cv2.imread(image)
#     if len(image.shape) == 3:
#         print("Found 3 Channels : {}".format(image.shape))
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         print("Converted to Gray Channel. Size : {}".format(image.shape))
#     else:
#         print("Image Shape : {}".format(image.shape))

#     print("Kernel Shape : {}".format(kernel.shape))

#     if verbose:
#         plt.imshow(image, cmap='gray')
#         plt.title("Image")
#         plt.show()

#     image_row, image_col = image.shape
#     kernel_row, kernel_col = kernel.shape

#     output = np.zeros(image.shape)

#     pad_height = int((kernel_row - 1) / 2)
#     pad_width = int((kernel_col - 1) / 2)

#     padded_image = np.zeros(
#         (image_row + (2 * pad_height), image_col + (2 * pad_width)))

#     padded_image[pad_height:padded_image.shape[0] - pad_height,
#                  pad_width:padded_image.shape[1] - pad_width] = image

#     if verbose:
#         plt.imshow(padded_image, cmap='gray')
#         plt.title("Padded Image")
#         plt.show()

#     for row in range(image_row):
#         for col in range(image_col):
#             output[row, col] = np.sum(
#                 kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
#             if average:
#                 output[row, col] /= kernel.shape[0] * kernel.shape[1]

#     print("Output Image size : {}".format(output.shape))

#     if verbose:
#         plt.imshow(output, cmap='gray')
#         plt.title("Output Image using {}X{} Kernel".format(
#             kernel_row, kernel_col))
#         plt.show()

#     return output
