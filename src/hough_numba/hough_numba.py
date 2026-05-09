import numpy as np
from numba import njit, prange


@njit(parallel=True)
def hough_lines_numba(edge_image, rho_res, theta_res):
    height, width = edge_image.shape
    diagonal = int(np.ceil(np.sqrt(height ** 2 + width ** 2)))

    num_rhos = int((2 * diagonal) / rho_res) + 1
    num_thetas = int(np.pi / theta_res)

    accumulator = np.zeros((num_rhos, num_thetas), dtype=np.int32)

    cos_t = np.empty(num_thetas)
    sin_t = np.empty(num_thetas)

    for theta_idx in range(num_thetas):
        theta = theta_idx * theta_res
        cos_t[theta_idx] = np.cos(theta)
        sin_t[theta_idx] = np.sin(theta)

    for y in prange(height):
        for x in range(width):
            if edge_image[y, x] > 0:
                for theta_idx in range(num_thetas):
                    rho = int(round(x * cos_t[theta_idx] + y * sin_t[theta_idx]))
                    rho_idx = int((rho + diagonal) / rho_res)

                    if 0 <= rho_idx < num_rhos:
                        accumulator[rho_idx, theta_idx] += 1

    rhos = np.empty(num_rhos)
    thetas = np.empty(num_thetas)

    for i in range(num_rhos):
        rhos[i] = -diagonal + i * rho_res

    for i in range(num_thetas):
        thetas[i] = i * theta_res

    return accumulator, rhos, thetas