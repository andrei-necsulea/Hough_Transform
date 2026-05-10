import tkinter as tk
from tkinter import filedialog, messagebox
import time
import os
import csv
import threading

import numpy as np
import cv2
import matplotlib.pyplot as plt

from hough_sequential.hough_core import hough_lines_sequential
from hough_numba.hough_numba import hough_lines_numba
from image_utils import load_grayscale_image, compute_edges, draw_detected_lines
from dataset_manager import DATASETS, download_limited_images, find_images_recursively


class HoughGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hough Transform - Line Detection")
        self.root.geometry("1050x850")
        self.root.resizable(True, True)

        self.image_path = None
        self.folder_path = None

        self.create_widgets()

    def create_widgets(self):
        title_label = tk.Label(
            self.root,
            text="Hough Transform - Line Detection",
            font=("Arial", 18, "bold")
        )
        title_label.pack(pady=10)

        mode_frame = tk.LabelFrame(self.root, text="Processing Mode", padx=15, pady=10)
        mode_frame.pack(pady=8)

        self.mode_var = tk.StringVar(value="Single Image")

        tk.Radiobutton(
            mode_frame,
            text="Single Image Mode",
            variable=self.mode_var,
            value="Single Image"
        ).grid(row=0, column=0, padx=10)

        tk.Radiobutton(
            mode_frame,
            text="Batch Folder Mode",
            variable=self.mode_var,
            value="Batch Folder"
        ).grid(row=0, column=1, padx=10)

        tk.Radiobutton(
            mode_frame,
            text="Predefined Dataset Mode",
            variable=self.mode_var,
            value="Predefined Dataset"
        ).grid(row=0, column=2, padx=10)

        implementation_frame = tk.LabelFrame(self.root, text="Implementation", padx=15, pady=10)
        implementation_frame.pack(pady=8)

        self.implementation_var = tk.StringVar(value="Sequential")

        implementation_menu = tk.OptionMenu(
            implementation_frame,
            self.implementation_var,
            "Sequential",
            "Numba Parallel"
        )
        implementation_menu.config(width=22)
        implementation_menu.pack()

        self.path_label = tk.Label(
            self.root,
            text="No image, folder or dataset selected",
            fg="gray",
            wraplength=850
        )
        self.path_label.pack(pady=5)

        select_frame = tk.Frame(self.root)
        select_frame.pack(pady=8)

        tk.Button(
            select_frame,
            text="Select Image",
            width=22,
            command=self.select_image
        ).grid(row=0, column=0, padx=10)

        tk.Button(
            select_frame,
            text="Select Folder",
            width=22,
            command=self.select_folder
        ).grid(row=0, column=1, padx=10)

        dataset_frame = tk.LabelFrame(
            self.root,
            text="Predefined Dataset",
            padx=15,
            pady=10
        )
        dataset_frame.pack(pady=8)

        self.dataset_var = tk.StringVar(value=list(DATASETS.keys())[0])

        dataset_menu = tk.OptionMenu(
            dataset_frame,
            self.dataset_var,
            *DATASETS.keys()
        )
        dataset_menu.config(width=40)
        dataset_menu.grid(row=0, column=0, padx=8, pady=5)

        tk.Label(dataset_frame, text="Max images:").grid(row=0, column=1, padx=8)

        self.max_images_entry = tk.Entry(dataset_frame, width=8)
        self.max_images_entry.insert(0, "20")
        self.max_images_entry.grid(row=0, column=2, padx=8)

        tk.Button(
            dataset_frame,
            text="Download / Load Dataset",
            width=22,
            command=self.start_download_dataset_thread
        ).grid(row=0, column=3, padx=8)

        params_frame = tk.LabelFrame(self.root, text="Hough Parameters", padx=15, pady=15)
        params_frame.pack(pady=8)

        tk.Label(params_frame, text="Rho resolution:").grid(row=0, column=0, padx=8, pady=6, sticky="e")
        self.rho_entry = tk.Entry(params_frame, width=12)
        self.rho_entry.insert(0, "1")
        self.rho_entry.grid(row=0, column=1, padx=8, pady=6)

        tk.Label(params_frame, text="Theta resolution (degrees):").grid(row=1, column=0, padx=8, pady=6, sticky="e")
        self.theta_entry = tk.Entry(params_frame, width=12)
        self.theta_entry.insert(0, "1")
        self.theta_entry.grid(row=1, column=1, padx=8, pady=6)

        tk.Label(params_frame, text="Line threshold:").grid(row=2, column=0, padx=8, pady=6, sticky="e")
        self.threshold_entry = tk.Entry(params_frame, width=12)
        self.threshold_entry.insert(0, "120")
        self.threshold_entry.grid(row=2, column=1, padx=8, pady=6)

        buttons_frame = tk.Frame(self.root)
        buttons_frame.pack(pady=12)

        tk.Button(
            buttons_frame,
            text="Run Selected Mode",
            width=24,
            bg="#2d89ef",
            fg="white",
            command=self.start_run_selected_mode_thread
        ).grid(row=0, column=0, padx=8, pady=5)

        tk.Button(
            buttons_frame,
            text="Benchmark Seq vs Numba",
            width=24,
            command=self.start_benchmark_thread
        ).grid(row=0, column=1, padx=8, pady=5)

        tk.Button(
            buttons_frame,
            text="Generate Comparison Chart",
            width=26,
            command=self.start_chart_thread
        ).grid(row=0, column=2, padx=8, pady=5)

        tk.Button(
            buttons_frame,
            text="Batch Benchmark Folder",
            width=24,
            command=self.start_batch_benchmark_thread
        ).grid(row=1, column=0, padx=8, pady=5)

        tk.Button(
            buttons_frame,
            text="Clear Results",
            width=24,
            command=self.clear_results
        ).grid(row=1, column=1, padx=8, pady=5)

        tk.Button(
            buttons_frame,
            text="Copy Results",
            width=24,
            command=self.copy_results
        ).grid(row=1, column=2, padx=8, pady=5)

        result_frame = tk.LabelFrame(self.root, text="Execution Results", padx=10, pady=10)
        result_frame.pack(pady=10, fill="both", expand=True, padx=30)

        self.result_text = tk.Text(
            result_frame,
            height=14,
            font=("Consolas", 10),
            wrap="word"
        )
        self.result_text.pack(fill="both", expand=True)

        self.write_results("No execution yet.")

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.image_path = file_path
            self.mode_var.set("Single Image")
            self.path_label.config(text=f"Selected image: {file_path}", fg="black")

    def select_folder(self):
        folder_path = filedialog.askdirectory(title="Select image folder")

        if folder_path:
            self.folder_path = folder_path
            self.mode_var.set("Batch Folder")
            self.path_label.config(text=f"Selected folder: {folder_path}", fg="black")

    def validate_parameters(self):
        try:
            rho_res = int(self.rho_entry.get())
            theta_degrees = float(self.theta_entry.get())
            threshold = int(self.threshold_entry.get())

            if rho_res <= 0:
                raise ValueError("Rho resolution must be greater than 0.")

            if theta_degrees <= 0:
                raise ValueError("Theta resolution must be greater than 0.")

            if threshold <= 0:
                raise ValueError("Threshold must be greater than 0.")

            theta_res = np.deg2rad(theta_degrees)

            return rho_res, theta_res, threshold

        except ValueError as error:
            messagebox.showerror("Invalid parameters", str(error))
            return None

    def get_max_images(self):
        try:
            max_images = int(self.max_images_entry.get())

            if max_images <= 0:
                raise ValueError

            return max_images

        except ValueError:
            messagebox.showerror("Invalid value", "Max images must be a positive integer.")
            return None

    def write_results(self, text):
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert("1.0", text)
        self.result_text.config(state="disabled")

    def safe_write_results(self, text):
        self.root.after(0, lambda: self.write_results(text))

    def safe_message_info(self, title, message):
        self.root.after(0, lambda: messagebox.showinfo(title, message))

    def safe_message_error(self, title, message):
        self.root.after(0, lambda: messagebox.showerror(title, message))

    def start_thread(self, target):
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()

    def start_download_dataset_thread(self):
        self.start_thread(self.download_predefined_dataset)

    def start_run_selected_mode_thread(self):
        self.start_thread(self.run_selected_mode)

    def start_benchmark_thread(self):
        self.start_thread(self.benchmark_current_mode)

    def start_chart_thread(self):
        self.start_thread(self.generate_chart_current_mode)

    def start_batch_benchmark_thread(self):
        self.start_thread(self.batch_benchmark_current_mode)

    def run_selected_mode(self):
        mode = self.mode_var.get()

        if mode == "Single Image":
            self.run_single_image()
        elif mode in ["Batch Folder", "Predefined Dataset"]:
            self.run_dataset_selected_implementation()

    def benchmark_current_mode(self):
        mode = self.mode_var.get()

        if mode == "Single Image":
            self.benchmark_single_image()
        elif mode in ["Batch Folder", "Predefined Dataset"]:
            self.benchmark_dataset()

    def generate_chart_current_mode(self):
        mode = self.mode_var.get()

        if mode == "Single Image":
            self.generate_single_image_chart()
        elif mode in ["Batch Folder", "Predefined Dataset"]:
            self.generate_dataset_chart()

    def batch_benchmark_current_mode(self):
        mode = self.mode_var.get()

        if mode == "Single Image":
            self.safe_message_error(
                "Invalid mode",
                "Batch Benchmark Folder works only in Batch Folder Mode or Predefined Dataset Mode."
            )
            return

        self.benchmark_dataset()

    def run_hough_algorithm(self, implementation, edges, rho_res, theta_res):
        if implementation == "Sequential":
            start_time = time.perf_counter()

            accumulator, rhos, thetas = hough_lines_sequential(
                edges,
                rho_res=rho_res,
                theta_res=theta_res
            )

            end_time = time.perf_counter()

        elif implementation == "Numba Parallel":
            hough_lines_numba(edges, rho_res, theta_res)

            start_time = time.perf_counter()

            accumulator, rhos, thetas = hough_lines_numba(
                edges,
                rho_res,
                theta_res
            )

            end_time = time.perf_counter()

        else:
            raise ValueError("Unknown implementation selected.")

        return accumulator, rhos, thetas, end_time - start_time

    def run_single_image(self):
        if self.image_path is None:
            self.safe_message_error("Error", "Please select an image first.")
            return

        parameters = self.validate_parameters()

        if parameters is None:
            return

        rho_res, theta_res, threshold = parameters
        implementation = self.implementation_var.get()

        try:
            image = load_grayscale_image(self.image_path)
            edges = compute_edges(image)

            accumulator, rhos, thetas, execution_time = self.run_hough_algorithm(
                implementation,
                edges,
                rho_res,
                theta_res
            )

            result_image, detected_lines = draw_detected_lines(
                image,
                accumulator,
                rhos,
                thetas,
                threshold
            )

            os.makedirs("results/single_image", exist_ok=True)

            output_name = f"{implementation.lower().replace(' ', '_')}_detected_lines.jpg"
            output_path = os.path.join("results", "single_image", output_name)

            cv2.imwrite(output_path, result_image)

            result_text = (
                f"Mode: Single Image\n"
                f"Implementation: {implementation}\n"
                f"Execution time: {execution_time:.6f} seconds\n"
                f"Image: {os.path.basename(self.image_path)}\n"
                f"Image size: {image.shape}\n"
                f"Edge pixels: {np.count_nonzero(edges)}\n"
                f"Accumulator size: {accumulator.shape}\n"
                f"Rho resolution: {rho_res}\n"
                f"Theta resolution: {np.rad2deg(theta_res):.2f} degrees\n"
                f"Line threshold: {threshold}\n"
                f"Max votes: {accumulator.max()}\n"
                f"Detected lines: {detected_lines}\n"
                f"Saved result: {output_path}"
            )

            self.safe_write_results(result_text)
            self.safe_message_info("Done", f"Hough Transform completed.\n\nResult saved in:\n{output_path}")

        except Exception as error:
            self.safe_message_error("Execution error", str(error))

    def get_current_dataset_images(self):
        if self.folder_path is None:
            self.safe_message_error(
                "Error",
                "Please select a folder or download/load a predefined dataset first."
            )
            return None

        max_images = self.get_max_images()

        if max_images is None:
            return None

        image_files = find_images_recursively(self.folder_path)

        if not image_files:
            self.safe_message_error("Error", "No image files found.")
            return None

        return image_files[:max_images]

    def run_dataset_selected_implementation(self):
        parameters = self.validate_parameters()

        if parameters is None:
            return

        image_files = self.get_current_dataset_images()

        if image_files is None:
            return

        rho_res, theta_res, threshold = parameters
        implementation = self.implementation_var.get()

        results = []
        output_dir = os.path.join("results", "dataset_outputs")
        os.makedirs(output_dir, exist_ok=True)

        try:
            for index, image_path in enumerate(image_files, start=1):
                self.safe_write_results(
                    f"Processing dataset with {implementation}...\n"
                    f"Image {index}/{len(image_files)}: {os.path.basename(image_path)}"
                )

                image = load_grayscale_image(image_path)
                edges = compute_edges(image)

                accumulator, rhos, thetas, execution_time = self.run_hough_algorithm(
                    implementation,
                    edges,
                    rho_res,
                    theta_res
                )

                result_image, detected_lines = draw_detected_lines(
                    image,
                    accumulator,
                    rhos,
                    thetas,
                    threshold
                )

                output_name = (
                    f"{os.path.splitext(os.path.basename(image_path))[0]}_"
                    f"{implementation.lower().replace(' ', '_')}.jpg"
                )

                output_path = os.path.join(output_dir, output_name)
                cv2.imwrite(output_path, result_image)

                results.append({
                    "image": os.path.basename(image_path),
                    "implementation": implementation,
                    "time": execution_time,
                    "edge_pixels": int(np.count_nonzero(edges)),
                    "height": image.shape[0],
                    "width": image.shape[1],
                    "accumulator_shape": str(accumulator.shape),
                    "max_votes": int(accumulator.max()),
                    "detected_lines": int(detected_lines),
                    "output_path": output_path
                })

            csv_path = os.path.join("results", "dataset_selected_implementation.csv")
            self.save_batch_csv(results, csv_path)

            total_time = sum(item["time"] for item in results)
            avg_time = total_time / len(results)

            result_text = (
                f"Mode: {self.mode_var.get()}\n"
                f"Dataset processing completed.\n"
                f"Implementation: {implementation}\n"
                f"Images processed: {len(results)}\n"
                f"Total execution time: {total_time:.6f} seconds\n"
                f"Average execution time: {avg_time:.6f} seconds/image\n"
                f"CSV saved: {csv_path}\n\n"
            )

            for item in results:
                result_text += (
                    f"{item['image']} | "
                    f"time={item['time']:.6f}s | "
                    f"edges={item['edge_pixels']} | "
                    f"lines={item['detected_lines']}\n"
                )

            self.safe_write_results(result_text)
            self.safe_message_info("Done", f"Dataset processing completed.\nCSV saved in:\n{csv_path}")

        except Exception as error:
            self.safe_message_error("Dataset processing error", str(error))

    def benchmark_single_image(self):
        if self.image_path is None:
            self.safe_message_error("Error", "Please select an image first.")
            return

        parameters = self.validate_parameters()

        if parameters is None:
            return

        rho_res, theta_res, threshold = parameters

        try:
            image = load_grayscale_image(self.image_path)
            edges = compute_edges(image)

            seq_acc, _, _, seq_time = self.run_hough_algorithm(
                "Sequential",
                edges,
                rho_res,
                theta_res
            )

            _, _, _, numba_time = self.run_hough_algorithm(
                "Numba Parallel",
                edges,
                rho_res,
                theta_res
            )

            speedup = seq_time / numba_time if numba_time > 0 else 0
            cpu_count = os.cpu_count() if os.cpu_count() else 1
            efficiency = speedup / cpu_count

            result_text = (
                "Benchmark: Sequential Hough vs Numba Parallel Hough\n"
                f"Mode: Single Image\n"
                f"Image: {os.path.basename(self.image_path)}\n"
                f"Image size: {image.shape}\n"
                f"Edge pixels: {np.count_nonzero(edges)}\n"
                f"Accumulator size: {seq_acc.shape}\n"
                f"Rho resolution: {rho_res}\n"
                f"Theta resolution: {np.rad2deg(theta_res):.2f} degrees\n\n"
                f"Sequential Hough time: {seq_time:.6f} seconds\n"
                f"Numba Parallel Hough time: {numba_time:.6f} seconds\n"
                f"Speedup: {speedup:.2f}x\n"
                f"Efficiency: {efficiency:.4f}\n"
                f"CPU cores detected: {cpu_count}"
            )

            self.safe_write_results(result_text)

        except Exception as error:
            self.safe_message_error("Benchmark error", str(error))

    def benchmark_dataset(self):
        parameters = self.validate_parameters()

        if parameters is None:
            return

        image_files = self.get_current_dataset_images()

        if image_files is None:
            return

        rho_res, theta_res, threshold = parameters
        benchmark_results = []

        try:
            for index, image_path in enumerate(image_files, start=1):
                self.safe_write_results(
                    f"Running dataset benchmark...\n"
                    f"Image {index}/{len(image_files)}: {os.path.basename(image_path)}"
                )

                image = load_grayscale_image(image_path)
                edges = compute_edges(image)

                seq_acc, _, _, seq_time = self.run_hough_algorithm(
                    "Sequential",
                    edges,
                    rho_res,
                    theta_res
                )

                _, _, _, numba_time = self.run_hough_algorithm(
                    "Numba Parallel",
                    edges,
                    rho_res,
                    theta_res
                )

                speedup = seq_time / numba_time if numba_time > 0 else 0
                cpu_count = os.cpu_count() if os.cpu_count() else 1
                efficiency = speedup / cpu_count

                benchmark_results.append({
                    "image": os.path.basename(image_path),
                    "height": image.shape[0],
                    "width": image.shape[1],
                    "edge_pixels": int(np.count_nonzero(edges)),
                    "accumulator_shape": str(seq_acc.shape),
                    "sequential_time": seq_time,
                    "numba_time": numba_time,
                    "speedup": speedup,
                    "efficiency": efficiency
                })

            os.makedirs("results", exist_ok=True)

            csv_path = os.path.join("results", "dataset_benchmark_seq_numba.csv")
            self.save_benchmark_csv(benchmark_results, csv_path)

            total_seq = sum(item["sequential_time"] for item in benchmark_results)
            total_numba = sum(item["numba_time"] for item in benchmark_results)

            total_speedup = total_seq / total_numba if total_numba > 0 else 0
            cpu_count = os.cpu_count() if os.cpu_count() else 1
            total_efficiency = total_speedup / cpu_count

            result_text = (
                f"Benchmark: Sequential Hough vs Numba Parallel Hough\n"
                f"Mode: {self.mode_var.get()}\n"
                f"Images processed: {len(benchmark_results)}\n"
                f"CSV saved: {csv_path}\n\n"
                f"Total Sequential time: {total_seq:.6f} seconds\n"
                f"Total Numba time: {total_numba:.6f} seconds\n"
                f"Total Speedup: {total_speedup:.2f}x\n"
                f"Total Efficiency: {total_efficiency:.4f}\n"
                f"CPU cores detected: {cpu_count}\n\n"
            )

            for item in benchmark_results:
                result_text += (
                    f"{item['image']} | "
                    f"Seq={item['sequential_time']:.6f}s | "
                    f"Numba={item['numba_time']:.6f}s | "
                    f"Speedup={item['speedup']:.2f}x | "
                    f"Efficiency={item['efficiency']:.4f}\n"
                )

            self.safe_write_results(result_text)

        except Exception as error:
            self.safe_message_error("Dataset benchmark error", str(error))

    def generate_single_image_chart(self):
        if self.image_path is None:
            self.safe_message_error("Error", "Please select an image first.")
            return

        parameters = self.validate_parameters()

        if parameters is None:
            return

        rho_res, theta_res, threshold = parameters

        try:
            image = load_grayscale_image(self.image_path)
            edges = compute_edges(image)

            _, _, _, seq_time = self.run_hough_algorithm(
                "Sequential",
                edges,
                rho_res,
                theta_res
            )

            _, _, _, numba_time = self.run_hough_algorithm(
                "Numba Parallel",
                edges,
                rho_res,
                theta_res
            )

            chart_path = os.path.join("results", "single_image_comparison_chart.png")

            self.create_comparison_chart(
                title="Single Image Benchmark",
                labels=["Sequential Hough", "Numba Parallel Hough"],
                sequential_times=[seq_time],
                numba_times=[numba_time],
                image_names=[os.path.basename(self.image_path)],
                output_path=chart_path,
                show_chart=True
            )

            speedup = seq_time / numba_time if numba_time > 0 else 0
            cpu_count = os.cpu_count() if os.cpu_count() else 1
            efficiency = speedup / cpu_count

            result_text = (
                "Single Image Chart generated.\n"
                f"Chart saved: {chart_path}\n\n"
                f"Sequential time: {seq_time:.6f} seconds\n"
                f"Numba time: {numba_time:.6f} seconds\n"
                f"Speedup: {speedup:.2f}x\n"
                f"Efficiency: {efficiency:.4f}"
            )

            self.safe_write_results(result_text)

        except Exception as error:
            self.safe_message_error("Chart error", str(error))

    def generate_dataset_chart(self):
        parameters = self.validate_parameters()

        if parameters is None:
            return

        image_files = self.get_current_dataset_images()

        if image_files is None:
            return

        rho_res, theta_res, threshold = parameters
        benchmark_results = []

        try:
            for index, image_path in enumerate(image_files, start=1):
                self.safe_write_results(
                    f"Generating dataset chart...\n"
                    f"Image {index}/{len(image_files)}: {os.path.basename(image_path)}"
                )

                image = load_grayscale_image(image_path)
                edges = compute_edges(image)

                _, _, _, seq_time = self.run_hough_algorithm(
                    "Sequential",
                    edges,
                    rho_res,
                    theta_res
                )

                _, _, _, numba_time = self.run_hough_algorithm(
                    "Numba Parallel",
                    edges,
                    rho_res,
                    theta_res
                )

                benchmark_results.append({
                    "image": os.path.basename(image_path),
                    "sequential_time": seq_time,
                    "numba_time": numba_time
                })

            image_names = [item["image"] for item in benchmark_results]
            sequential_times = [item["sequential_time"] for item in benchmark_results]
            numba_times = [item["numba_time"] for item in benchmark_results]

            chart_path = os.path.join("results", "dataset_comparison_chart.png")

            self.create_comparison_chart(
                title=f"{self.mode_var.get()} Benchmark",
                labels=["Sequential Hough", "Numba Parallel Hough"],
                sequential_times=sequential_times,
                numba_times=numba_times,
                image_names=image_names,
                output_path=chart_path,
                show_chart=True
            )

            total_seq = sum(sequential_times)
            total_numba = sum(numba_times)
            speedup = total_seq / total_numba if total_numba > 0 else 0
            cpu_count = os.cpu_count() if os.cpu_count() else 1
            efficiency = speedup / cpu_count

            result_text = (
                f"Dataset Chart generated.\n"
                f"Mode: {self.mode_var.get()}\n"
                f"Images processed: {len(image_names)}\n"
                f"Chart saved: {chart_path}\n\n"
                f"Total Sequential time: {total_seq:.6f} seconds\n"
                f"Total Numba time: {total_numba:.6f} seconds\n"
                f"Total Speedup: {speedup:.2f}x\n"
                f"Total Efficiency: {efficiency:.4f}"
            )

            self.safe_write_results(result_text)

        except Exception as error:
            self.safe_message_error("Dataset chart error", str(error))

    def create_comparison_chart(
        self,
        title,
        labels,
        sequential_times,
        numba_times,
        image_names,
        output_path,
        show_chart=True
    ):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        total_seq = sum(sequential_times)
        total_numba = sum(numba_times)

        speedup = total_seq / total_numba if total_numba > 0 else 0
        cpu_count = os.cpu_count() if os.cpu_count() else 1
        efficiency = speedup / cpu_count

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        implementations = labels
        total_times = [total_seq, total_numba]

        bars = axes[0, 0].bar(implementations, total_times)
        axes[0, 0].set_title("Total Execution Time")
        axes[0, 0].set_ylabel("Seconds")

        for bar, value in zip(bars, total_times):
            axes[0, 0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}s",
                ha="center",
                va="bottom"
            )

        x = np.arange(len(image_names))
        width = 0.35

        axes[0, 1].bar(x - width / 2, sequential_times, width, label="Sequential")
        axes[0, 1].bar(x + width / 2, numba_times, width, label="Numba")
        axes[0, 1].set_title("Execution Time per Image")
        axes[0, 1].set_ylabel("Seconds")
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(image_names, rotation=30, ha="right")
        axes[0, 1].legend()

        bars_speedup = axes[1, 0].bar(["Speedup"], [speedup])
        axes[1, 0].set_title("Overall Speedup")
        axes[1, 0].set_ylabel("x")

        for bar in bars_speedup:
            axes[1, 0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{speedup:.2f}x",
                ha="center",
                va="bottom"
            )

        bars_eff = axes[1, 1].bar(["Efficiency"], [efficiency])
        axes[1, 1].set_title("Overall Efficiency")
        axes[1, 1].set_ylabel("Efficiency")

        for bar in bars_eff:
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{efficiency:.4f}",
                ha="center",
                va="bottom"
            )

        fig.suptitle(title, fontsize=16)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)

        if show_chart:
            plt.show()
        else:
            plt.close()

    def download_predefined_dataset(self):
        dataset_name = self.dataset_var.get()
        max_images = self.get_max_images()

        if max_images is None:
            return

        try:
            self.safe_write_results(
                f"Downloading only {max_images} images from:\n"
                f"{dataset_name}\n\n"
                "This uses Kaggle API file-by-file download."
            )

            dataset_path = download_limited_images(dataset_name, max_images)

            self.folder_path = dataset_path
            self.mode_var.set("Predefined Dataset")

            image_files = find_images_recursively(dataset_path)

            self.root.after(
                0,
                lambda: self.path_label.config(
                    text=f"Loaded dataset subset: {dataset_name}\nPath: {dataset_path}",
                    fg="black"
                )
            )

            result_text = (
                f"Dataset subset downloaded successfully.\n"
                f"Dataset: {dataset_name}\n"
                f"Path: {dataset_path}\n"
                f"Requested images: {max_images}\n"
                f"Images found locally: {len(image_files)}\n\n"
                f"All mode buttons now operate on this predefined dataset."
            )

            self.safe_write_results(result_text)

        except Exception as error:
            self.safe_message_error("Dataset error", str(error))

    def save_batch_csv(self, results, csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        with open(csv_path, "w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "image",
                    "implementation",
                    "time",
                    "edge_pixels",
                    "height",
                    "width",
                    "accumulator_shape",
                    "max_votes",
                    "detected_lines",
                    "output_path"
                ]
            )

            writer.writeheader()
            writer.writerows(results)

    def save_benchmark_csv(self, results, csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        with open(csv_path, "w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "image",
                    "height",
                    "width",
                    "edge_pixels",
                    "accumulator_shape",
                    "sequential_time",
                    "numba_time",
                    "speedup",
                    "efficiency"
                ]
            )

            writer.writeheader()
            writer.writerows(results)

    def clear_results(self):
        self.write_results("No execution yet.")

    def copy_results(self):
        text = self.result_text.get("1.0", tk.END).strip()

        if not text or text == "No execution yet.":
            messagebox.showwarning("Warning", "There are no results to copy.")
            return

        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()

        messagebox.showinfo("Copied", "Results copied to clipboard.")


if __name__ == "__main__":
    root = tk.Tk()
    app = HoughGUI(root)
    root.mainloop()