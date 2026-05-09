import time
import numpy as np
import cv2
import os

from hough_numba.hough_numba import hough_lines_numba
from image_utils import load_grayscale_image, compute_edges, draw_detected_lines


def main():
    image_path = "data/images/h1.png"

    if not os.path.exists(image_path):
        print("Image not found. Add test.jpg in data/images/")
        return

    image = load_grayscale_image(image_path)
    edges = compute_edges(image)

    rho_res = 1
    theta_res = np.pi / 180
    threshold = 120

    print("Running first compilation pass...")
    hough_lines_numba(edges, rho_res, theta_res)

    print("Running measured pass...")
    start = time.perf_counter()
    accumulator, rhos, thetas = hough_lines_numba(edges, rho_res, theta_res)
    end = time.perf_counter()

    os.makedirs("results", exist_ok=True)
    result_image, detected_lines = draw_detected_lines(
        image,
        accumulator,
        rhos,
        thetas,
        threshold
    )

    output_path = "results/numba_detected_lines.jpg"
    cv2.imwrite(output_path, result_image)

    print("\n===== Numba Parallel Hough Transform =====")
    print(f"Image size: {image.shape}")
    print(f"Edge pixels: {np.count_nonzero(edges)}")
    print(f"Accumulator size: {accumulator.shape}")
    print(f"Execution time: {end - start:.6f} seconds")
    print(f"Maximum votes: {accumulator.max()}")
    print(f"Detected lines: {detected_lines}")
    print(f"Saved result: {output_path}")


if __name__ == "__main__":
    main()