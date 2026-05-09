import tkinter as tk
from tkinter import filedialog, messagebox
import time
import os
import numpy as np
import cv2

from hough_sequential.hough_core import hough_lines_sequential
from image_utils import load_grayscale_image, compute_edges, draw_detected_lines
from hough_numba.hough_numba import hough_lines_numba


class HoughGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hough Transform - Line Detection")
        self.root.geometry("760x650")
        self.root.resizable(True, True)

        self.image_path = None

        self.create_widgets()

    def create_widgets(self):
        title_label = tk.Label(
            self.root,
            text="Hough Transform - Line Detection",
            font=("Arial", 18, "bold")
        )
        title_label.pack(pady=15)

        implementation_frame = tk.LabelFrame(
            self.root,
            text="Implementation",
            padx=15,
            pady=10
        )
        implementation_frame.pack(pady=10)

        self.implementation_var = tk.StringVar()
        self.implementation_var.set("Sequential")

        implementation_menu = tk.OptionMenu(
            implementation_frame,
            self.implementation_var,
            "Sequential",
            "Numba Parallel"
        )

        implementation_menu.config(width=20)
        implementation_menu.pack()

        self.path_label = tk.Label(
            self.root,
            text="No image selected",
            fg="gray",
            wraplength=620
        )
        self.path_label.pack(pady=5)

        select_button = tk.Button(
            self.root,
            text="Select Image",
            width=25,
            command=self.select_image
        )
        select_button.pack(pady=10)

        params_frame = tk.LabelFrame(
            self.root,
            text="Hough Parameters",
            padx=15,
            pady=15
        )
        params_frame.pack(pady=10)

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
        buttons_frame.pack(pady=15)

        run_button = tk.Button(
            buttons_frame,
            text="Run Selected Implementation",
            width=25,
            bg="#2d89ef",
            fg="white",
            command=self.run_hough
        )
        run_button.grid(row=0, column=0, padx=10)

        benchmark_button = tk.Button(
            buttons_frame,
            text="Benchmark Seq vs Numba",
            width=22,
            command=self.benchmark_seq_numba
        )
        benchmark_button.grid(row=1, column=0, columnspan=3, pady=10)

        clear_button = tk.Button(
            buttons_frame,
            text="Clear Results",
            width=18,
            command=self.clear_results
        )
        clear_button.grid(row=0, column=1, padx=10)

        copy_button = tk.Button(
            buttons_frame,
            text="Copy Results",
            width=18,
            command=self.copy_results
        )
        copy_button.grid(row=0, column=2, padx=10)

        result_frame = tk.LabelFrame(
            self.root,
            text="Execution Results",
            padx=10,
            pady=10
        )
        result_frame.pack(pady=10, fill="both", expand=True, padx=30)

        self.result_text = tk.Text(
            result_frame,
            height=12,
            font=("Consolas", 10),
            wrap="word"
        )
        self.result_text.pack(fill="both", expand=True)

        self.result_text.insert("1.0", "No execution yet.")
        self.result_text.config(state="disabled")

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
            self.path_label.config(text=file_path, fg="black")

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

    def run_hough(self):
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

            if implementation == "Sequential":

                start_time = time.perf_counter()

                accumulator, rhos, thetas = hough_lines_sequential(
                    edges,
                    rho_res=rho_res,
                    theta_res=theta_res
                )

                end_time = time.perf_counter()

            elif implementation == "Numba Parallel":

                # warm-up compilation
                hough_lines_numba(edges, rho_res, theta_res)

                start_time = time.perf_counter()

                accumulator, rhos, thetas = hough_lines_numba(
                    edges,
                    rho_res,
                    theta_res
                )

                end_time = time.perf_counter()

            execution_time = end_time - start_time

            result_image, detected_lines = draw_detected_lines(
                            image,
                            accumulator,
                            rhos,
                            thetas,
                            threshold
          )

            os.makedirs("results", exist_ok=True)

            output_path = os.path.join("results", f"{implementation.lower().replace(' ', '_')}_detected_lines.jpg")
            cv2.imwrite(output_path, result_image)

            result_text = (
                f"Implementation: {implementation}\n"
                f"Execution time: {execution_time:.6f} seconds\n"
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

            self.result_text.config(state="normal")
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert("1.0", result_text)
            self.result_text.config(state="disabled")

            messagebox.showinfo(
                "Done",
                f"Hough Transform completed.\n\nResult saved in:\n{output_path}"
            )

        except Exception as error:
            messagebox.showerror("Execution error", str(error))
    


    def benchmark_seq_numba(self):
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

            # Numba warm-up
            hough_lines_numba(edges, rho_res, theta_res)

            start_seq = time.perf_counter()
            seq_acc, seq_rhos, seq_thetas = hough_lines_sequential(
                edges,
                rho_res=rho_res,
                theta_res=theta_res
            )
            end_seq = time.perf_counter()

            start_numba = time.perf_counter()
            numba_acc, numba_rhos, numba_thetas = hough_lines_numba(
                edges,
                rho_res,
                theta_res
            )
            end_numba = time.perf_counter()

            seq_time = end_seq - start_seq
            numba_time = end_numba - start_numba

            speedup = seq_time / numba_time if numba_time > 0 else 0
            efficiency = speedup / os.cpu_count() if os.cpu_count() else 0

            result_text = (
                "Benchmark: Sequential vs Numba Parallel\n"
                f"Image size: {image.shape}\n"
                f"Edge pixels: {np.count_nonzero(edges)}\n"
                f"Accumulator size: {seq_acc.shape}\n"
                f"Rho resolution: {rho_res}\n"
                f"Theta resolution: {np.rad2deg(theta_res):.2f} degrees\n"
                f"Sequential time: {seq_time:.6f} seconds\n"
                f"Numba time: {numba_time:.6f} seconds\n"
                f"Speedup: {speedup:.2f}x\n"
                f"Efficiency: {efficiency:.4f}\n"
                f"CPU cores detected: {os.cpu_count()}"
            )

            self.result_text.config(state="normal")
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert("1.0", result_text)
            self.result_text.config(state="disabled")

        except Exception as error:
            messagebox.showerror("Benchmark error", str(error))



    def clear_results(self):
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert("1.0", "No execution yet.")
        self.result_text.config(state="disabled")
    
    def copy_results(self):

        text = self.result_text.get("1.0", tk.END).strip()

        if not text or text == "No execution yet.":
            messagebox.showwarning(
                "Warning",
                "There are no results to copy."
            )
            return

        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()

        messagebox.showinfo(
            "Copied",
            "Results copied to clipboard."
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = HoughGUI(root)
    root.mainloop()