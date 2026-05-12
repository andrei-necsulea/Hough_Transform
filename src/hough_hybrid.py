from mpi4py import MPI
from numba import njit, prange
import numpy as np
import cv2
import os
import time
import sys

from image_utils import load_grayscale_image, compute_edges, draw_detected_lines


@njit(parallel=True)
def compute_local_accumulator_numba(y_idxs, x_idxs, rhos_len, thetas, diagonal, rho_res):
    accumulator = np.zeros((rhos_len, len(thetas)), dtype=np.int32)

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    for i in prange(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for theta_idx in range(len(thetas)):
            rho = int(round(x * cos_t[theta_idx] + y * sin_t[theta_idx]))
            rho_idx = int((rho + diagonal) / rho_res)

            if 0 <= rho_idx < rhos_len:
                accumulator[rho_idx, theta_idx] += 1

    return accumulator


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/images/h1.png"
    rho_res = 1
    theta_res = np.pi / 180
    threshold = 120

    image_shape = None
    y_idxs = None
    x_idxs = None

    if rank == 0:
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            comm.Abort()

        image = load_grayscale_image(image_path)
        edges = compute_edges(image)

        y_idxs, x_idxs = np.nonzero(edges)
        image_shape = edges.shape

        print("===== Hybrid MPI + Numba Hough Transform =====")
        print(f"MPI processes: {size}")
        print(f"Image: {image_path}")
        print(f"Image size: {image_shape}")
        print(f"Edge pixels: {len(x_idxs)}")

    image_shape = comm.bcast(image_shape, root=0)

    height, width = image_shape
    diagonal = int(np.ceil(np.sqrt(height ** 2 + width ** 2)))

    rhos = np.arange(-diagonal, diagonal + 1, rho_res)
    thetas = np.arange(0, np.pi, theta_res)

    rhos_len = len(rhos)

    if rank == 0:
        y_chunks = np.array_split(y_idxs, size)
        x_chunks = np.array_split(x_idxs, size)
    else:
        y_chunks = None
        x_chunks = None

    local_y_idxs = comm.scatter(y_chunks, root=0)
    local_x_idxs = comm.scatter(x_chunks, root=0)

    # Warm-up Numba compilation before timing
    compute_local_accumulator_numba(
        local_y_idxs,
        local_x_idxs,
        rhos_len,
        thetas,
        diagonal,
        rho_res
    )

    comm.Barrier()
    start_time = time.perf_counter()

    local_accumulator = compute_local_accumulator_numba(
        local_y_idxs,
        local_x_idxs,
        rhos_len,
        thetas,
        diagonal,
        rho_res
    )

    global_accumulator = np.zeros_like(local_accumulator)

    comm.Reduce(
        [local_accumulator, MPI.INT],
        [global_accumulator, MPI.INT],
        op=MPI.SUM,
        root=0
    )

    comm.Barrier()
    end_time = time.perf_counter()

    if rank == 0:
        execution_time = end_time - start_time

        os.makedirs("results", exist_ok=True)

        result_image, detected_lines = draw_detected_lines(
            image,
            global_accumulator,
            rhos,
            thetas,
            threshold
        )

        output_path = "results/hybrid_mpi_numba_detected_lines.jpg"
        cv2.imwrite(output_path, result_image)

        print(f"Accumulator size: {global_accumulator.shape}")
        print(f"Execution time: {execution_time:.6f} seconds")
        print(f"Max votes: {global_accumulator.max()}")
        print(f"Detected lines: {detected_lines}")
        print(f"Saved result: {output_path}")


if __name__ == "__main__":
    main()