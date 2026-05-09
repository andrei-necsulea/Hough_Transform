import cv2
import numpy as np
import time
import os


def hough_lines_sequential(edge_image, rho_res=1, theta_res=np.pi / 180):

    height, width = edge_image.shape

    diagonal = int(np.ceil(np.sqrt(height**2 + width**2)))

    rhos = np.arange(-diagonal, diagonal + 1, rho_res)

    thetas = np.arange(0, np.pi, theta_res)

    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int32)

    y_idxs, x_idxs = np.nonzero(edge_image)

    for i in range(len(x_idxs)):

        x = x_idxs[i]
        y = y_idxs[i]

        for theta_idx in range(len(thetas)):

            theta = thetas[theta_idx]

            rho = int(round(x * np.cos(theta) + y * np.sin(theta)))

            rho_idx = rho + diagonal

            accumulator[rho_idx, theta_idx] += 1

    return accumulator, rhos, thetas


def main():

    image_path = "data/images/h1.png"

    if not os.path.exists(image_path):
        print("Image not found.")
        return

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Could not load image.")
        return

    edges = cv2.Canny(image, 50, 150)

    start_time = time.time()

    accumulator, rhos, thetas = hough_lines_sequential(edges)

    end_time = time.time()

    execution_time = end_time - start_time

    print("\n===== Sequential Hough Transform =====")
    print(f"Image size: {image.shape}")
    print(f"Accumulator size: {accumulator.shape}")
    print(f"Execution time: {execution_time:.6f} seconds")
    print(f"Maximum votes: {accumulator.max()}")


if __name__ == "__main__":
    main()