import tkinter as tk
from tkinter import filedialog, messagebox
import time
import os
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt

from hough_sequential.hough_core import hough_lines_sequential
from image_utils import load_grayscale_image, compute_edges, draw_detected_lines
from hough_numba.hough_numba import hough_lines_numba


class HoughGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hough Transform - Line Detection")
        self.root.geometry("850x750")
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
            text="No image or folder selected",
            fg="gray",
            wraplength=760
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
            command=self.run_selected_mode
        ).grid(row=0, column=0, padx=8, pady=5)

        tk.Button(
            buttons_frame,
            text="Benchmark Seq vs Numba",
            width=24,
            command=self.benchmark_seq_numba_single
        ).grid(row=0, column=1, padx=8, pady=5)

        tk.Button(
            buttons_frame,
            text="Generate Comparison Chart",
            width=26,
            command=self.generate_comparison_chart_single
        ).grid(row=0, column=2, padx=8, pady=5)

        tk.Button(
            buttons_frame,
            text="Batch Benchmark Folder",
            width=24,
            command=self.batch_benchmark_folder
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

    def write_results(self, text):
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert("1.0", text)
        self.result_text.config(state="disabled")

    def run_selected_mode(self):
        mode = self.mode_var.get()

        if mode == "Single Image":
            self.run_single_image()
        else:
            self.run_batch_selected_implementation()

    def run_single_image(self):
        if self.image_path is None:
            messagebox.showerror("Error", "Please select an image first.")
            return

        parameters = self.validate_parameters()
        if parameters is None:
            return

        rho_res, theta_res, threshold = parameters

        try:
            image = load_grayscale_image(self.image_path)
            edges = compute_edges(image)
            implementation = self.implementation_var.get()

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

            os.makedirs("results", exist_ok=True)

            output_name = f"{implementation.lower().replace(' ', '_')}_detected_lines.jpg"
            output_path = os.path.join("results", output_name)
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

            self.write_results(result_text)

            messagebox.showinfo(
                "Done",
                f"Hough Transform completed.\n\nResult saved in:\n{output_path}"
            )

        except Exception as error:
            messagebox.showerror("Execution error", str(error))

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

    def benchmark_seq_numba_single(self):
        if self.image_path is None:
            messagebox.showerror("Error", "Please select an image first.")
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

            numba_acc, _, _, numba_time = self.run_hough_algorithm(
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

            self.write_results(result_text)

        except Exception as error:
            messagebox.showerror("Benchmark error", str(error))

    def generate_comparison_chart_single(self):
        if self.image_path is None:
            messagebox.showerror("Error", "Please select an image first.")
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

            self.create_comparison_chart(
                image_name=os.path.basename(self.image_path),
                image_shape=image.shape,
                edge_pixels=np.count_nonzero(edges),
                seq_time=seq_time,
                numba_time=numba_time,
                output_path=os.path.join("results", "single_image_benchmark_comparison.png"),
                show_chart=True
            )

        except Exception as error:
            messagebox.showerror("Chart error", str(error))

    def run_batch_selected_implementation(self):
        if self.folder_path is None:
            messagebox.showerror("Error", "Please select a folder first.")
            return

        parameters = self.validate_parameters()
        if parameters is None:
            return

        rho_res, theta_res, threshold = parameters
        implementation = self.implementation_var.get()

        image_files = self.get_image_files(self.folder_path)

        if not image_files:
            messagebox.showerror("Error", "No image files found in selected folder.")
            return

        results = []

        try:
            os.makedirs("results/batch_outputs", exist_ok=True)

            for image_path in image_files:
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

                output_path = os.path.join("results", "batch_outputs", output_name)
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
                    "detected_lines": detected_lines,
                    "output_path": output_path
                })

            csv_path = os.path.join("results", "batch_selected_implementation.csv")
            self.save_batch_csv(results, csv_path)

            result_text = f"Batch processing completed.\nImplementation: {implementation}\nImages processed: {len(results)}\nCSV saved: {csv_path}\n\n"

            for item in results:
                result_text += (
                    f"{item['image']} | "
                    f"time={item['time']:.6f}s | "
                    f"edges={item['edge_pixels']} | "
                    f"lines={item['detected_lines']}\n"
                )

            self.write_results(result_text)

            messagebox.showinfo("Done", f"Batch processing completed.\nCSV saved in:\n{csv_path}")

        except Exception as error:
            messagebox.showerror("Batch error", str(error))

    def batch_benchmark_folder(self):
        if self.folder_path is None:
            messagebox.showerror("Error", "Please select a folder first.")
            return

        parameters = self.validate_parameters()
        if parameters is None:
            return

        rho_res, theta_res, threshold = parameters
        image_files = self.get_image_files(self.folder_path)

        if not image_files:
            messagebox.showerror("Error", "No image files found in selected folder.")
            return

        benchmark_results = []

        try:
            for image_path in image_files:
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

            csv_path = os.path.join("results", "batch_benchmark_seq_numba.csv")
            self.save_benchmark_csv(benchmark_results, csv_path)

            chart_path = os.path.join("results", "batch_benchmark_comparison.png")
            self.create_batch_chart(benchmark_results, chart_path)

            result_text = (
                "Batch benchmark completed.\n"
                f"Images processed: {len(benchmark_results)}\n"
                f"CSV saved: {csv_path}\n"
                f"Chart saved: {chart_path}\n\n"
            )

            for item in benchmark_results:
                result_text += (
                    f"{item['image']} | "
                    f"Seq={item['sequential_time']:.6f}s | "
                    f"Numba={item['numba_time']:.6f}s | "
                    f"Speedup={item['speedup']:.2f}x | "
                    f"Efficiency={item['efficiency']:.4f}\n"
                )

            self.write_results(result_text)

            messagebox.showinfo(
                "Done",
                f"Batch benchmark completed.\n\nCSV: {csv_path}\nChart: {chart_path}"
            )

        except Exception as error:
            messagebox.showerror("Batch benchmark error", str(error))

    def create_comparison_chart(
        self,
        image_name,
        image_shape,
        edge_pixels,
        seq_time,
        numba_time,
        output_path,
        show_chart=True
    ):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        cpu_count = os.cpu_count() if os.cpu_count() else 1

        numba_speedup = seq_time / numba_time if numba_time > 0 else 0
        numba_efficiency = numba_speedup / cpu_count

        implementations = ["Sequential Hough", "Numba Parallel Hough"]
        execution_times = [seq_time, numba_time]
        speedups = [1.0, numba_speedup]
        efficiencies = [1.0, numba_efficiency]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        bars1 = axes[0].bar(implementations, execution_times)
        axes[0].set_title("Execution Time")
        axes[0].set_ylabel("Seconds")

        for bar, value in zip(bars1, execution_times):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}s",
                ha="center",
                va="bottom"
            )

        bars2 = axes[1].bar(implementations, speedups)
        axes[1].set_title("Speedup")
        axes[1].set_ylabel("x")

        for bar, value in zip(bars2, speedups):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}x",
                ha="center",
                va="bottom"
            )

        bars3 = axes[2].bar(implementations, efficiencies)
        axes[2].set_title("Efficiency")
        axes[2].set_ylabel("Efficiency")

        for bar, value in zip(bars3, efficiencies):
            axes[2].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom"
            )

        fig.suptitle(
            "Hough Transform Benchmark: Sequential vs Numba Parallel\n"
            f"Image: {image_name} | Size: {image_shape} | "
            f"Edge pixels: {edge_pixels} | CPU cores: {cpu_count}",
            fontsize=14
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)

        if show_chart:
            plt.show()
        else:
            plt.close()

        result_text = (
            "Single image comparison chart generated.\n"
            f"Chart saved: {output_path}\n\n"
            f"Sequential Hough time: {seq_time:.6f} seconds\n"
            f"Numba Parallel Hough time: {numba_time:.6f} seconds\n"
            f"Speedup: {numba_speedup:.2f}x\n"
            f"Efficiency: {numba_efficiency:.4f}\n"
            f"CPU cores detected: {cpu_count}"
        )

        self.write_results(result_text)

    def create_batch_chart(self, benchmark_results, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        image_names = [item["image"] for item in benchmark_results]
        sequential_times = [item["sequential_time"] for item in benchmark_results]
        numba_times = [item["numba_time"] for item in benchmark_results]
        speedups = [item["speedup"] for item in benchmark_results]
        efficiencies = [item["efficiency"] for item in benchmark_results]

        x = np.arange(len(image_names))
        width = 0.35

        fig, axes = plt.subplots(3, 1, figsize=(14, 14))

        axes[0].bar(x - width / 2, sequential_times, width, label="Sequential Hough")
        axes[0].bar(x + width / 2, numba_times, width, label="Numba Parallel Hough")
        axes[0].set_title("Execution Time per Image")
        axes[0].set_ylabel("Seconds")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(image_names, rotation=30, ha="right")
        axes[0].legend()

        axes[1].bar(image_names, speedups)
        axes[1].set_title("Speedup per Image")
        axes[1].set_ylabel("Speedup")
        axes[1].tick_params(axis="x", rotation=30)

        axes[2].bar(image_names, efficiencies)
        axes[2].set_title("Efficiency per Image")
        axes[2].set_ylabel("Efficiency")
        axes[2].tick_params(axis="x", rotation=30)

        fig.suptitle("Batch Benchmark: Sequential Hough vs Numba Parallel Hough", fontsize=15)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.show()

    def get_image_files(self, folder_path):
        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")

        return [
            os.path.join(folder_path, file_name)
            for file_name in os.listdir(folder_path)
            if file_name.lower().endswith(valid_extensions)
        ]

    def save_batch_csv(self, results, csv_path):
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