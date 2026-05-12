import subprocess
import time
import csv
import os
import sys
import re
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hough_sequential.hough_core import hough_lines_sequential
from hough_numba.hough_numba import hough_lines_numba
from image_utils import load_grayscale_image, compute_edges


def extract_execution_time(output):
    match = re.search(r"Execution time:\s*([0-9.]+)", output)

    if match:
        return float(match.group(1))

    return None


def run_python_benchmark(image_path, rho_res, theta_res):
    image = load_grayscale_image(image_path)
    edges = compute_edges(image)

    start = time.perf_counter()
    seq_acc, _, _ = hough_lines_sequential(edges, rho_res, theta_res)
    end = time.perf_counter()
    sequential_time = end - start

    hough_lines_numba(edges, rho_res, theta_res)

    start = time.perf_counter()
    numba_acc, _, _ = hough_lines_numba(edges, rho_res, theta_res)
    end = time.perf_counter()
    numba_time = end - start

    return {
        "edge_pixels": int(np.count_nonzero(edges)),
        "accumulator_shape": str(seq_acc.shape),
        "sequential_time": sequential_time,
        "numba_time": numba_time
    }


def run_mpi_script(script_path, processes):
    command = [
        "mpiexec",
        "-n",
        str(processes),
        sys.executable,
        script_path
    ]

    env = os.environ.copy()
    env["NUMBA_NUM_THREADS"] = "2"
    env["OMP_NUM_THREADS"] = "2"

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=env,
        timeout=120
    )

    output = result.stdout + result.stderr
    execution_time = extract_execution_time(output)

    if execution_time is None:
        print(output)
        raise RuntimeError(f"Could not extract execution time from {script_path}")

    return execution_time


def main():
    image_path = "data/images/h1.png"

    rho_res = 1
    theta_res = np.pi / 180
    processes_list = [2, 4]

    os.makedirs("results", exist_ok=True)

    base_results = run_python_benchmark(image_path, rho_res, theta_res)

    sequential_time = base_results["sequential_time"]
    numba_time = base_results["numba_time"]

    rows = []

    rows.append({
        "implementation": "Sequential",
        "processes": 1,
        "threads": 1,
        "edge_pixels": base_results["edge_pixels"],
        "accumulator_shape": base_results["accumulator_shape"],
        "execution_time": sequential_time,
        "speedup": 1.0,
        "efficiency": 1.0
    })

    rows.append({
        "implementation": "Numba Parallel",
        "processes": 1,
        "threads": os.cpu_count(),
        "edge_pixels": base_results["edge_pixels"],
        "accumulator_shape": base_results["accumulator_shape"],
        "execution_time": numba_time,
        "speedup": sequential_time / numba_time,
        "efficiency": (sequential_time / numba_time) / os.cpu_count()
    })

    for processes in processes_list:
        mpi_time = run_mpi_script("src/hough_mpi.py", processes)

        rows.append({
            "implementation": "MPI",
            "processes": processes,
            "threads": 1,
            "edge_pixels": base_results["edge_pixels"],
            "accumulator_shape": base_results["accumulator_shape"],
            "execution_time": mpi_time,
            "speedup": sequential_time / mpi_time,
            "efficiency": (sequential_time / mpi_time) / processes
        })

        hybrid_time = run_mpi_script("src/hough_hybrid.py", processes)

        rows.append({
            "implementation": "Hybrid MPI + Numba",
            "processes": processes,
            "threads": os.cpu_count(),
            "edge_pixels": base_results["edge_pixels"],
            "accumulator_shape": base_results["accumulator_shape"],
            "execution_time": hybrid_time,
            "speedup": sequential_time / hybrid_time,
            "efficiency": (sequential_time / hybrid_time) / processes
        })

    csv_path = "results/benchmark_all.csv"

    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "implementation",
                "processes",
                "threads",
                "edge_pixels",
                "accumulator_shape",
                "execution_time",
                "speedup",
                "efficiency"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\n===== Final Benchmark Results =====")

    for row in rows:
        print(
            f"{row['implementation']} | "
            f"processes={row['processes']} | "
            f"threads={row['threads']} | "
            f"time={row['execution_time']:.6f}s | "
            f"speedup={row['speedup']:.2f}x | "
            f"efficiency={row['efficiency']:.4f}"
        )

    print(f"\nSaved CSV: {csv_path}")


if __name__ == "__main__":
    main()