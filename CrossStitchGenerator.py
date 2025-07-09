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
import math
from PIL import ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from sklearn.cluster import KMeans
import numpy as np


class CrossStitchApp:
    def __init__(self, root):
        self.root = root  # self is a class level identifier
        self.root.title("Cross Stitch Chart Generator")

        self.image_path = None
        self.quantized_image = None

        # File selection
        self.select_button = tk.Button(root, text="Select Image",
                                       command=self.select_file)  # Select_button calls select_file function which is defined in the CrossStitchApp class
        self.select_button.pack(pady=10)

        # Fabric count size inputs
        self.count_label = tk.Label(root,
                                    text="Fabric count (stitches per inch)")  # add function to switch between cm and inches later on
        self.count_label.pack()
        self.count = tk.Entry(root, width=5)
        self.count.insert(0, "14")
        self.count.pack()

        # Grid size inputs
        self.grid_label = tk.Label(root, text="Canvas Dimensions(inches)")
        self.grid_label.pack()
        self.grid_width = tk.Entry(root, width=5)
        self.grid_width.insert(0, "5")
        self.grid_width.pack()
        self.grid_height = tk.Entry(root, width=5)
        self.grid_height.insert(0, "5")
        self.grid_height.pack()

        # Number of colors input
        self.color_label = tk.Label(root, text="Number of Colors")
        self.color_label.pack()
        self.num_colors = tk.Entry(root, width=5)
        self.num_colors.insert(0, "10")
        self.num_colors.pack()

        # Sharpen input
        self.sharpen_label = tk.Label(root, text="Image sharpening")
        self.sharpen_label.pack()
        self.sharpen = tk.Entry(root, width=5)
        self.sharpen.insert(0, "True")
        self.sharpen.pack()

        # Contrast input
        self.contrast_label = tk.Label(root, text="Image Contrast")
        self.contrast_label.pack()
        self.contrast = tk.Entry(root, width=5)
        self.contrast.insert(0, "1")
        self.contrast.pack()

        # Generate button
        self.generate_button = tk.Button(root, text="Generate Chart", command=self.generate_chart)
        self.generate_button.pack(pady=20)

        # Feedback
        self.status_label = tk.Label(root, text="", fg="blue")
        self.status_label.pack()

    def select_file(self):
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        if file_path:
            self.image_path = file_path
            self.status_label.config(text=f"Selected: {file_path}")

    def quantize_image(self, img, num_colors, contrast, sharpen):

        # Step 1: Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)

        # Step 2: Optional sharpening
        if sharpen:
            img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=100, threshold=2))

        img_np = np.array(img)
        h, w, _ = img_np.shape
        flat_img = img_np.reshape(-1, 3)

        #Avoid clustering errors if unique colours is less than user specified colours
        unique_colors = np.unique(flat_img, axis=0)
        if len(unique_colors) < num_colors:
            num_colors = len(unique_colors)

        kmeans = KMeans(n_clusters=num_colors, random_state=0).fit(flat_img)
        new_colors = kmeans.cluster_centers_.astype('uint8')
        labels = kmeans.labels_

        quantized = new_colors[labels].reshape(h, w, 3)
        return Image.fromarray(quantized), new_colors

    def generate_chart(self):
        #Define list of synbols to be used in cross stitch chart
        SYMBOLS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()<>?/|+=~")

        if not self.image_path:
            messagebox.showerror("Error", "Please select an image first.")
            return

        try:
            count = int(self.count.get())
            width = int(self.grid_width.get())  # Combine stitch count and final canvas size to get grid dimensions
            height = int(self.grid_height.get())
            width_st_count = width * count
            height_st_count = height * count
            num_colors = int(self.num_colors.get())
            sharpen = self.sharpen.get().lower() == "true"
            contrast = float(self.contrast.get())

            # Load and resize image
            img = Image.open(self.image_path).convert('RGB')
            img = img.resize((width_st_count, height_st_count),
                             Image.NEAREST)  # Current display is approx. 94 pixel per inch.  Print scaling line/offer user to set scale?

            # Quantize colors
            quant_img, palette = self.quantize_image(img, num_colors, contrast, sharpen)

            # Convert quantized image to array for symbol mapping
            img_array = np.array(quant_img)
            h, w, _ = img_array.shape

            # Match each color to a symbol
            color_palette = [tuple(color) for color in palette]
            symbol_map = {tuple(color_palette[i]): SYMBOLS[i % len(SYMBOLS)] for i in range(len(color_palette))}

            self.status_label.config(
                text=f"Generated for {width}in x {height}in chart on {count} thread count with {num_colors} colors.")

        except Exception as e:
            messagebox.showerror("Error", str(e))

        # Ask user where to save the chart
        save_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF Files", "*.pdf"), ("PNG Files", "*.png")],
            title="Save Chart As"
        )

        if not save_path:
            return  # User cancelled save dialog

        # Plot the image with grid overlay
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(quant_img)
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.set_aspect('equal')

        # Light grid lines (every stitch)
        ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
        ax.grid(which='minor', color='lightgray', linestyle=':', linewidth=0.5)

        # Bold grid every 10 stitches
        for x in range(0, w + 1, 10):
            ax.axvline(x - 0.5, color='black', linewidth=1)
        for y in range(0, h + 1, 10):
            ax.axhline(y - 0.5, color='black', linewidth=1)

        # Overlay symbols
        for y in range(h):
            for x in range(w):
                rgb = tuple(img_array[y, x])
                symbol = symbol_map.get(rgb, '?')
                ax.text(x, y, symbol, fontsize=6, ha='center', va='center', color='black')

        # Hide axes
        ax.set_xticks([])
        ax.set_yticks([])

        # Color key legend
        legend_fig, legend_ax = plt.subplots(figsize=(5, len(color_palette) * 0.4))
        legend_ax.axis('off')
        for i, color in enumerate(color_palette):
            symbol = symbol_map[tuple(color)]
            patch_color = np.array(color) / 255
            legend_ax.text(0, i, f"{symbol}", fontsize=10, ha='left', va='center')
            legend_ax.add_patch(plt.Rectangle((0.2, i - 0.4), 0.4, 0.8, color=patch_color))
            legend_ax.text(0.7, i, f"RGB: {color}", fontsize=8, va='center')

        # Save chart and legend
        chart_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF", "*.pdf")],
                                                  title="Save Chart")
        if not chart_path:
            return

        fig.savefig(chart_path, dpi=300, bbox_inches='tight')
        legend_path = chart_path.replace(".pdf", "_legend.pdf")
        legend_fig.savefig(legend_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        plt.close(legend_fig)

        self.status_label.config(text=f"Chart saved to:\n{chart_path}\nand legend:\n{legend_path}")

        plt.show()

        # Save to file
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)

        self.status_label.config(text=f"Chart saved to: {save_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CrossStitchApp(root)  # Call cross stitch generator program which is contained within its own class
    root.mainloop()  # This keeps GUI running infinitely