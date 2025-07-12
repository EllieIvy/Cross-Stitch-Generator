# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 16:55:34 2025

@author: ellie
"""

#--------------------------------------------------------------------------------
#Cross Stitch Chart Generator
#User selects image to be transformed into a cross stitch chart
#User selects the number of colours to be used
#User selects grid resolution
#Sharpness and contrast can be adjusted
#Output 1 is scaled vision of what the image will look like with corresponding colours
#Output 2 is colour chart for cross stitch (needs to be big enough that instructions are visible)
#--------------------------------------------------------------------------------
import io
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
from PIL.Image import Resampling
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import string


# ---------- Image Processing Functions ----------
def open_image():
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
    if path:
        return Image.open(path).convert("RGB"), path
    return None, None


def quantize_image(img, width, height, num_colors, contrast=1.5, sharpen=True):
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast)

    if sharpen:
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

    img = img.resize((width, height), Image.NEAREST)
    img_np = np.array(img)
    h, w, _ = img_np.shape
    flat_img = img_np.reshape(-1, 3)

    kmeans = KMeans(n_clusters=num_colors, random_state=0).fit(flat_img)
    palette = kmeans.cluster_centers_.astype("uint8")
    labels = kmeans.labels_

    quantized_img = palette[labels].reshape(h, w, 3)
    return Image.fromarray(quantized_img), labels.reshape(h, w), palette


def overlay_grid_symbols(image, labels, palette):
    draw = ImageDraw.Draw(image)
    h, w = labels.shape
    font = ImageFont.load_default()

    symbols = list(string.ascii_letters + string.digits + string.punctuation)
    while len(symbols) < len(palette):
        symbols += symbols

    symbol_map = {i: symbols[i] for i in range(len(palette))}

    cell_width = image.width // w
    cell_height = image.height // h

    for y in range(h):
        for x in range(w):
            label = labels[y, x]
            sym = symbol_map[label]
            text_x = x * cell_width + cell_width // 4
            text_y = y * cell_height + cell_height // 4
            draw.text((text_x, text_y), sym, fill="black", font=font)

    return image


def save_chart_with_legend(quant_img, labels, palette):
    symbols = list(string.ascii_letters + string.digits + string.punctuation)
    while len(symbols) < len(palette):
        symbols += symbols
    symbol_map = {i: symbols[i] for i in range(len(palette))}

    h, w = labels.shape

    save_path = filedialog.asksaveasfilename(
        defaultextension=".pdf",
        filetypes=[("PDF Files", "*.pdf"), ("PNG Files", "*.png")],
        title="Save Chart As"
    )
    if not save_path:
        return

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(quant_img)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_aspect('equal')

    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which='minor', color='lightgray', linestyle=':', linewidth=0.5)

    for x in range(0, w + 1, 10):
        ax.axvline(x - 0.5, color='black', linewidth=1)
    for y in range(0, h + 1, 10):
        ax.axhline(y - 0.5, color='black', linewidth=1)

    for y in range(h):
        for x in range(w):
            label = labels[y, x]
            symbol = symbol_map[label]
            ax.text(x, y, symbol, fontsize=6, ha='center', va='center', color='black')

    ax.set_xticks([])
    ax.set_yticks([])

    legend_fig, legend_ax = plt.subplots(figsize=(5, len(palette) * 0.4))
    legend_ax.axis('off')
    for i, color in enumerate(palette):
        symbol = symbol_map[i]
        patch_color = np.array(color) / 255
        legend_ax.text(0, i, f"{symbol}", fontsize=10, ha='left', va='center')
        legend_ax.add_patch(plt.Rectangle((0.2, i - 0.4), 0.4, 0.8, color=patch_color))
        legend_ax.text(0.7, i, f"RGB: {tuple(color)}", fontsize=8, va='center')

    chart_path = save_path
    fig.savefig(chart_path, dpi=300, bbox_inches='tight')
    legend_path = chart_path.replace(".pdf", "_legend.pdf")
    legend_fig.savefig(legend_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    plt.close(legend_fig)


# ---------- GUI Application ----------
class CrossStitchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cross Stitch Chart Generator")

        self.original_image = None
        self.image_path = None

        self.setup_ui()

    def setup_ui(self):
        self.root.geometry("1200x700")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.grid(row=0, column=0, sticky="ns")

        ttk.Button(control_frame, text="Open Image", command=self.load_image).pack(fill="x")

        ttk.Label(control_frame, text="Fabric Count (threads per inch)").pack()
        self.fabric_entry = ttk.Entry(control_frame)
        self.fabric_entry.insert(0, "14")
        self.fabric_entry.pack(fill="x")

        ttk.Label(control_frame, text="Width (inches)").pack()
        self.inch_width_entry = ttk.Entry(control_frame)
        self.inch_width_entry.insert(0, "4")
        self.inch_width_entry.pack(fill="x")

        ttk.Label(control_frame, text="Height (inches)").pack()
        self.inch_height_entry = ttk.Entry(control_frame)
        self.inch_height_entry.insert(0, "4")
        self.inch_height_entry.pack(fill="x")

        ttk.Label(control_frame, text="Number of Colors").pack()
        self.color_slider = tk.Scale(control_frame, from_=2, to=30, orient='horizontal')
        self.color_slider.set(10)
        self.color_slider.pack(fill="x")

        ttk.Label(control_frame, text="Contrast").pack()
        self.contrast_slider = tk.Scale(control_frame, from_=0.5, to=3.0, resolution=0.1, orient='horizontal')
        self.contrast_slider.set(1.5)
        self.contrast_slider.pack(fill="x")

        self.sharpen_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Sharpen Image", variable=self.sharpen_var).pack()

        ttk.Button(control_frame, text="Update Preview", command=self.update_preview).pack(pady=10, fill="x")
        ttk.Button(control_frame, text="Export Chart", command=self.export_chart).pack(pady=10, fill="x")

        self.preview_frame = ttk.Frame(self.root)
        self.preview_frame.grid(row=0, column=1, sticky="nsew")
        self.preview_frame.grid_rowconfigure(1, weight=1)
        self.preview_frame.grid_columnconfigure(0, weight=1)

        self.original_canvas = None
        self.processed_canvas = None

    def show_image(self, img, title, row, max_display_size=(400, 400)):
        # Resize image to fit within max_display_size, preserving aspect ratio
        img_ratio = img.width / img.height
        max_w, max_h = max_display_size

        if img.width > max_w or img.height > max_h:
            if img_ratio > 1:
                # Wider than tall
                new_w = max_w
                new_h = int(max_w / img_ratio)
            else:
                # Taller than wide
                new_h = max_h
                new_w = int(max_h * img_ratio)
            img = img.resize((new_w, new_h), Image.NEAREST)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

        canvas = FigureCanvasTkAgg(fig, master=self.preview_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=row, column=0)

        if row == 0:
            if self.original_canvas:
                self.original_canvas.get_tk_widget().destroy()
            self.original_canvas = canvas
        else:
            if self.processed_canvas:
                self.processed_canvas.get_tk_widget().destroy()
            self.processed_canvas = canvas

    def load_image(self):
        img, path = open_image()
        if img:
            self.original_image = img
            self.image_path = path
            # Show original smaller (max 400x400)
            self.show_image(img, title="Original Image", row=0, max_display_size=(400, 400))
            self.update_preview()

    def update_preview(self):
        if not self.original_image:
            return

        try:
            fabric_count = float(self.fabric_entry.get())
            inch_width = float(self.inch_width_entry.get())
            inch_height = float(self.inch_height_entry.get())

            if fabric_count <= 0 or inch_width <= 0 or inch_height <= 0:
                raise ValueError

            width = int(fabric_count * inch_width)
            height = int(fabric_count * inch_height)
        except ValueError:
            messagebox.showerror("Invalid Input", "Enter positive numbers for fabric count and dimensions.")
            return

        colors = self.color_slider.get()
        contrast = self.contrast_slider.get()
        sharpen = self.sharpen_var.get()

        processed, labels, palette = quantize_image(
            self.original_image, width, height, colors, contrast=contrast, sharpen=sharpen
        )

        self.latest_quantized = processed
        self.latest_labels = labels
        self.latest_palette = palette

        # Show processed bigger (max 600x600), without grid overlay or symbols
        self.show_image(processed, title="Processed Preview", row=1, max_display_size=(600, 600))

    def export_chart(self):
        if hasattr(self, 'latest_quantized') and hasattr(self, 'latest_labels') and hasattr(self, 'latest_palette'):
            save_chart_with_legend(self.latest_quantized, self.latest_labels, self.latest_palette)
        else:
            messagebox.showerror("Error", "Please update the preview first before exporting.")


if __name__ == "__main__":
    root = tk.Tk()
    app = CrossStitchApp(root)
    root.mainloop()
