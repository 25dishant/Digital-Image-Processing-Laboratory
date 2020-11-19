import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


# A function to detect the hough space of the image


def HoughSpace(EdgeImage, num_rhos=180, num_thetas=180, t_count=220):
    edge_height, edge_width = EdgeImage.shape[:2]
    edge_height_half, edge_width_half = edge_height / 2, edge_width / 2

    d = np.sqrt(np.square(edge_height) + np.square(edge_width))
    dtheta = 180 / num_thetas
    drho = (2 * d) / num_rhos

    thetas = np.arange(0, 180, step=dtheta)
    rhos = np.arange(-d, d, step=drho)

    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))

    accumulator = np.zeros((len(rhos), len(rhos)))

    figure = plt.figure()
    subplot = figure.add_subplot()
    subplot.set_facecolor((0, 0, 0))

    for y in range(edge_height):
        for x in range(edge_width):
            if (EdgeImage[y][x].any() != 0):
                edge_point = [y - edge_height_half, x - edge_width_half]
                ys, xs = [], []
                for theta_idx in range(len(thetas)):
                    rho = (edge_point[1] * cos_thetas[theta_idx]) + \
                        (edge_point[0] * sin_thetas[theta_idx])
                    theta = thetas[theta_idx]
                    rho_idx = np.argmin(np.abs(rhos - rho))
                    accumulator[rho_idx][theta_idx] += 1
                    ys.append(rho)
                    xs.append(theta)
                subplot.plot(xs, ys, color="white", alpha=0.05)

    for y in range(accumulator.shape[0]):
        for x in range(accumulator.shape[1]):
            if accumulator[y][x] > t_count:
                rho = rhos[y]
                theta = thetas[x]
                a = np.cos(np.deg2rad(theta))
                b = np.sin(np.deg2rad(theta))
                x0 = (a * rho) + edge_width_half
                y0 = (b * rho) + edge_height_half
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                subplot.plot([theta], [rho], marker='.', color="red")

    subplot.invert_yaxis()
    subplot.invert_xaxis()

    subplot.title.set_text("Hough Space")
    plt.show()
    return accumulator, rhos, thetas


if __name__ == "__main__":
    EdgeImage = cv.imread("EdgeImage.jpg")
    EdgeImage = cv.dilate(
        EdgeImage,
        cv.getStructuringElement(cv.MORPH_RECT, (3, 3)),
        iterations=1)
    EdgeImage = cv.erode(
        EdgeImage,
        cv.getStructuringElement(cv.MORPH_RECT, (3, 3)),
        iterations=1)

    print("Please keep patience...This function takes a little time...")
    HoughSpace(EdgeImage)
