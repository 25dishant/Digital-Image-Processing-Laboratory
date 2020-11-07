"""
Non maximum suppression without interpolation requires us to divide the 3x3 grid of pixels into 8 sections. ie. if the gradient direction falls in between the angle -22.5 and 22.5, then we use the pixels that fall between this angle (r and q) as the value to compare with pixel p, see image below.

"""
import numpy as np


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
