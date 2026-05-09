import cv2
import numpy as np


def load_grayscale_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Could not read image.")

    return image


def compute_edges(image, low_threshold=50, high_threshold=150):
    return cv2.Canny(image, low_threshold, high_threshold)


def draw_detected_lines(original_image, accumulator, rhos, thetas, threshold):
    output = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

    line_indices = np.argwhere(accumulator >= threshold)

    for rho_idx, theta_idx in line_indices:
        rho = rhos[rho_idx]
        theta = thetas[theta_idx]

        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a * rho
        y0 = b * rho

        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)

        cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 1)

    return output, len(line_indices)