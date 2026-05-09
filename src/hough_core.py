import numpy as np


def hough_lines_sequential(edge_image, rho_res=1, theta_res=np.pi / 180):
    height, width = edge_image.shape
    diagonal = int(np.ceil(np.sqrt(height ** 2 + width ** 2)))

    rhos = np.arange(-diagonal, diagonal + 1, rho_res)
    thetas = np.arange(0, np.pi, theta_res)

    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int32)

    y_idxs, x_idxs = np.nonzero(edge_image)

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for theta_idx in range(len(thetas)):
            rho = int(round(x * cos_t[theta_idx] + y * sin_t[theta_idx]))
            rho_idx = rho + diagonal
            accumulator[rho_idx, theta_idx] += 1

    return accumulator, rhos, thetas