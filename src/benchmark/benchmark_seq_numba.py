import time
import os
import csv
import numpy as np

from hough_sequential.hough_core import hough_lines_sequential
from hough_numba.hough_numba import hough_lines_numba
from image_utils import load_grayscale_image, compute_edges


def measure_function(func, edges, rho_res, theta_res, runs=3):
    times = []

    for _ in range(runs):
        start = time.perf_counter()
        accumulator, rhos, thetas = func(edges, rho_res, theta_res)
        end = time.perf_counter()
        times.append(end - start)

    return min(times), accumulator


def main():
    image_path = "data/images/h1.png"
    rho_res = 1
    theta_res = np.pi / 180
    runs = 3

    os.makedirs("results", exist_ok=True)

    image = load_grayscale_image(image_path)
    edges = compute_edges(image)

    print("Warming up Numba...")
    hough_lines_numba(edges, rho_res, theta_res)

    seq_time, seq_acc = measure_function(
        hough_lines_sequential,
        edges,
        rho_res,
        theta_res,
        runs
    )

    numba_time, numba_acc = measure_function(
        hough_lines_numba,
        edges,
        rho_res,
        theta_res,
        runs
    )

    speedup = seq_time / numba_time
    efficiency = speedup / os.cpu_count()

    output_path = "results/benchmark_seq_numba.csv"

    with open(output_path, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "implementation",
            "image",
            "edge_pixels",
            "rho_res",
            "theta_res_degrees",
            "accumulator_shape",
            "execution_time",
            "speedup",
            "efficiency"
        ])

        writer.writerow([
            "Sequential",
            image_path,
            np.count_nonzero(edges),
            rho_res,
            np.rad2deg(theta_res),
            seq_acc.shape,
            seq_time,
            1.0,
            1.0
        ])

        writer.writerow([
            "Numba",
            image_path,
            np.count_nonzero(edges),
            rho_res,
            np.rad2deg(theta_res),
            numba_acc.shape,
            numba_time,
            speedup,
            efficiency
        ])

    print("\n===== Benchmark Sequential vs Numba =====")
    print(f"Image: {image_path}")
    print(f"Edge pixels: {np.count_nonzero(edges)}")
    print(f"Sequential time: {seq_time:.6f} s")
    print(f"Numba time: {numba_time:.6f} s")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Efficiency: {efficiency:.4f}")
    print(f"Saved CSV: {output_path}")


if __name__ == "__main__":
    main()