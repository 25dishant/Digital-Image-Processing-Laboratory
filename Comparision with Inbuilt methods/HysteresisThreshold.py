import numpy as np


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
