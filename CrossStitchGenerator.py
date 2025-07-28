#--------------------------------------------------------------------------------
# CROSS STITCH CHART GENERATOR
# User selects image to be transformed into a cross stitch chart
# User specifies canvas size, canvas thread count, number of colours to be used
# Image can be sharpened; contrast can be adjusted
# PDF Output 1: Cross stitch chart
# PDF Output 2: Colour key (specific to DMC thread colours)
#--------------------------------------------------------------------------------

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import string

# DMC Colour Palette
DMC_PALETTE = [
    {"code": "310", "name": "Black", "rgb": (0, 0, 0)},
    {"code": "321", "name": "Red", "rgb": (199, 43, 59)},
    {"code": "498", "name": "Dark Red", "rgb": (167, 19, 43)},
    {"code": "666", "name": "Bright Red", "rgb": (227, 29, 66)},
    {"code": "699", "name": "Green", "rgb": (0, 91, 0)},
    {"code": "703", "name": "Chartreuse", "rgb": (123, 181, 71)},
    {"code": "726", "name": "Topaz Light", "rgb": (255, 241, 148)},
    {"code": "742", "name": "Tangerine Light", "rgb": (255, 191, 87)},
    {"code": "743", "name": "Yellow Medium", "rgb": (254, 211, 118)},
    {"code": "754", "name": "Peach Light", "rgb": (247, 203, 191)},
    {"code": "796", "name": "Royal Blue Dark", "rgb": (17, 65, 109)},
    {"code": "798", "name": "Delft Blue Dark", "rgb": (70, 106, 142)},
    {"code": "799", "name": "Delft Blue Medium", "rgb": (116, 142, 182)},
    {"code": "820", "name": "Royal Blue Very Dark", "rgb": (14, 54, 92)},
    {"code": "900", "name": "Burnt Orange Dark", "rgb": (209, 88, 7)},
    {"code": "906", "name": "Parrot Green Medium", "rgb": (127, 179, 53)},
    {"code": "907", "name": "Parrot Green Light", "rgb": (199, 230, 102)},
    {"code": "909", "name": "Emerald Green Very Dark", "rgb": (21, 111, 73)},
    {"code": "934", "name": "Black Avocado Green", "rgb": (49, 57, 25)},
    {"code": "938", "name": "Coffee Brown Ultra Dark", "rgb": (54, 31, 14)},
    {"code": "939", "name": "Navy Blue Very Dark", "rgb": (27, 40, 83)},
    {"code": "945", "name": "Flesh Medium", "rgb": (247, 191, 169)},
    {"code": "959", "name": "Sea Green Medium", "rgb": (89, 199, 180)},
    {"code": "963", "name": "Dusty Rose Ultra Very Light", "rgb": (255, 215, 215)},
    {"code": "970", "name": "Pumpkin Light", "rgb": (247, 139, 19)},
    {"code": "995", "name": "Electric Blue Dark", "rgb": (0, 124, 146)},
    {"code": "3011", "name": "Khaki Green Dark", "rgb": (91, 98, 63)},
    {"code": "3021", "name": "Brown Gray Very Dark", "rgb": (79, 75, 65)},
    {"code": "3041", "name": "Antique Violet Medium", "rgb": (149, 111, 124)},
    {"code": "3072", "name": "Beaver Gray Very Light", "rgb": (230, 232, 232)},
    {"code": "3340", "name": "Apricot Medium", "rgb": (255, 131, 111)},
    {"code": "3341", "name": "Apricot", "rgb": (252, 171, 152)},
    {"code": "3607", "name": "Plum Light", "rgb": (197, 73, 137)},
    {"code": "3608", "name": "Plum Very Light", "rgb": (234, 156, 196)},
    {"code": "3685", "name": "Mauve Very Dark", "rgb": (136, 21, 49)},
    {"code": "3750", "name": "Antique Blue Very Dark", "rgb": (56, 76, 94)},
    {"code": "3760", "name": "Wedgewood Medium", "rgb": (62, 133, 162)},
    {"code": "3771", "name": "Terra Cotta Very Light", "rgb": (244, 187, 169)},
    {"code": "3822", "name": "Straw Light", "rgb": (246, 220, 152)},
    {"code": "3823", "name": "Yellow Ultra Pale", "rgb": (255, 253, 227)},
    {"code": "3829", "name": "Old Gold Very Dark", "rgb": (169, 130, 4)},
    {"code": "3837", "name": "Lavender Ultra Dark", "rgb": (108, 58, 110)},
    {"code": "3843", "name": "Electric Blue", "rgb": (20, 170, 208)},
    {"code": "3865", "name": "Winter White", "rgb": (249, 247, 241)},
    {"code": "5200", "name": "Snow White", "rgb": (255, 255, 255)}
]

#Match colours to DMC colours
def find_nearest_dmc_colour(colour_rgb):
    color_array = np.array(colour_rgb)
    distances = [np.linalg.norm(color_array - np.array(dmc["rgb"])) for dmc in DMC_PALETTE]
    return DMC_PALETTE[np.argmin(distances)]


# ---------- Image Processing Functions ----------
def open_image():
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
    if path:
        return Image.open(path).convert("RGB"), path
    return None, None

#Quantize image to reduce number of colours in cross stitch based on user-defined number of colours
def quantize_image(img, width, height, num_colours, contrast=1.5, sharpen=True):
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast)

    #Sharpen image if user has selected this option
    if sharpen:
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=100, threshold=0))

    #Store image as numpy array
    img = img.resize((width, height), Image.Resampling.NEAREST)
    img_np = np.array(img)
    h, w, _ = img_np.shape
    flat_img = img_np.reshape(-1, 3)

    #Use k-means clustering to find dominant colours
    kmeans: KMeans = KMeans(n_clusters=num_colours, random_state=0)
    kmeans.fit(flat_img)
    palette = kmeans.cluster_centers_.astype("uint8")
    labels = kmeans.labels_

    quantized_img = palette[labels].reshape(h, w, 3)
    return Image.fromarray(quantized_img), labels.reshape(h, w), palette

# Overlay colour key symbols on each grid position
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

# Create colour chart and colour key files
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

    # Get canvas width in inches from user input (default to 4 if not accessible)
    try:
        inch_width = float(app.inch_width_entry.get())
    except (ValueError, AttributeError):
        inch_width = 4.0

    # Adjust grid thickness and font size based on canvas size
    grid_linewidth = 2 / inch_width
    bold_grid_linewidth = 4 / inch_width
    font_size = 24 / inch_width

    # Main chart
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(quant_img)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_aspect('equal')

    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which='minor', color='lightgray', linestyle=':', linewidth=grid_linewidth)

    for x in range(0, w + 1, 10):
        ax.axvline(x - 0.5, color='black', linewidth=bold_grid_linewidth)
    for y in range(0, h + 1, 10):
        ax.axhline(y - 0.5, color='black', linewidth=bold_grid_linewidth)

    for y in range(h):
        for x in range(w):
            label = labels[y, x]
            symbol = symbol_map[label]
            r, g, b = palette[label]
            brightness = (0.299 * r + 0.587 * g + 0.114 * b)
            text_colour = 'white' if brightness < 128 else 'black'
            ax.text(x, y, symbol, fontsize=font_size, ha='center', va='center', color=text_colour)

    ax.set_xticks([])
    ax.set_yticks([])

    chart_path = save_path
    fig.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Legend chart
    legend_fig, legend_ax = plt.subplots(figsize=(5, len(palette) * 0.4))
    legend_ax.axis('off')

    for i, colour in enumerate(palette):
        patch_colour = np.array(colour) / 255
        symbol = symbol_map[i]
        dmc = find_nearest_dmc_colour(colour)
        y_pos = len(palette) - 1 - i
        legend_ax.add_patch(plt.Rectangle((0.05, y_pos - 0.4), 0.3, 0.8, color=patch_colour))
        legend_ax.text(0.4, y_pos, f"{symbol}", fontsize=10, ha='left', va='center')
        legend_ax.text(0.5, y_pos, f"DMC {dmc['code']} ({dmc['name']})", fontsize=8, ha='left', va='center')

    legend_ax.set_xlim(0, 1.2)
    legend_ax.set_ylim(-0.5, len(palette) - 0.5)

    legend_path = chart_path.replace(".pdf", "_legend.pdf")
    legend_fig.savefig(legend_path, dpi=300, bbox_inches='tight')
    plt.close(legend_fig)


# ---------- GUI Application ----------
class CrossStitchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cross Stitch Chart Generator")

        self.original_image = None
        self.image_path = None
        self.original_aspect_ratio = None

        self.fabric_entry = None
        self.keep_aspect_var = None
        self.inch_width_entry = None
        self.inch_height_entry = None
        self.colour_slider = None
        self.contrast_slider = None
        self.sharpen_var = None

        self.preview_frame = None
        self.original_canvas = None
        self.processed_canvas = None
        self.palette_frame = None

        self.latest_palette = None
        self.latest_quantized = None
        self.latest_labels = None

        self.setup_ui()

    def setup_ui(self):
        self.root.geometry("1200x600")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.grid(row=0, column=0, sticky="ns")

        ttk.Button(control_frame, text="Open Image", command=self.load_image).pack(pady=10, fill="x")

        ttk.Label(control_frame, text="Fabric Count (threads per inch)").pack(pady=10)
        self.fabric_entry = ttk.Entry(control_frame)
        self.fabric_entry.insert(0, "14")
        self.fabric_entry.pack(fill="x")

        self.keep_aspect_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="Maintain Aspect Ratio (only adjust width)",
                        variable=self.keep_aspect_var).pack(pady=15)

        ttk.Label(control_frame, text="Canvas Width (Inches)").pack(pady=10)
        self.inch_width_entry = ttk.Entry(control_frame)
        self.inch_width_entry.insert(0, "4")
        self.inch_width_entry.pack(fill="x")

        ttk.Label(control_frame, text="Canvas Height (Inches)").pack(pady=10)
        self.inch_height_entry = ttk.Entry(control_frame)
        self.inch_height_entry.insert(0, "4")
        self.inch_height_entry.pack(fill="x")

        ttk.Label(control_frame, text="Number of Colours").pack(pady=10)
        self.colour_slider = tk.Scale(control_frame, from_=2, to=30, orient='horizontal')
        self.colour_slider.set(10)
        self.colour_slider.pack(fill="x")

        ttk.Label(control_frame, text="Contrast").pack()
        self.contrast_slider = tk.Scale(control_frame, from_=0.5, to=3.0, resolution=0.1, orient='horizontal')
        self.contrast_slider.set(1.5)
        self.contrast_slider.pack(fill="x")

        self.sharpen_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Sharpen Image", variable=self.sharpen_var).pack(pady=10)

        ttk.Button(control_frame, text="Update Preview", command=self.update_preview).pack(pady=10, fill="x")
        ttk.Button(control_frame, text="Export Chart", command=self.export_chart).pack(pady=10, fill="x")

        self.preview_frame = ttk.Frame(self.root)
        self.preview_frame.grid(row=0, column=1, sticky="nsew")
        self.preview_frame.grid_rowconfigure(0, weight=1)
        self.preview_frame.grid_columnconfigure(0, weight=1)
        self.preview_frame.grid_columnconfigure(1, weight=1)

        self.original_canvas = None
        self.processed_canvas = None

    def load_image(self):
        img, path = open_image()
        if img:
            self.original_image = img
            self.image_path = path
            self.original_aspect_ratio = img.width / img.height
            self.show_image(img, title="Original Image", row=0)
            self.update_preview()

    def show_image(self, img, title, row):
        if title == "Original Image":
            fig, ax = plt.subplots(figsize=(4, 4))  # smaller original
        else:
            fig, ax = plt.subplots(figsize=(8, 8))  # larger preview

        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

        canvas = FigureCanvasTkAgg(fig, master=self.preview_frame)
        canvas.draw()

        col = 0 if title == "Original Image" else 1
        canvas.get_tk_widget().grid(row=0, column=col, sticky="nsew")

        if title == "Original Image":
            if self.original_canvas:
                self.original_canvas.get_tk_widget().destroy()
            self.original_canvas = canvas
        else:
            if self.processed_canvas:
                self.processed_canvas.get_tk_widget().destroy()
            self.processed_canvas = canvas

    # Update preview button to implement any changes in user-defined parameters
    def update_preview(self):
        if not self.original_image:
            return

        try:
            fabric_count = float(self.fabric_entry.get())
            inch_width = float(self.inch_width_entry.get())

            if fabric_count <= 0 or inch_width <= 0:
                raise ValueError

            width = int(fabric_count * inch_width)

            if self.keep_aspect_var.get() and self.original_aspect_ratio:
                height_inches = inch_width / self.original_aspect_ratio
                height = int(fabric_count * height_inches)
                self.inch_height_entry.delete(0, tk.END)
                self.inch_height_entry.insert(0, f"{height_inches:.2f}")
            else:
                inch_height = float(self.inch_height_entry.get())
                if inch_height <= 0:
                    raise ValueError
                height = int(fabric_count * inch_height)
        except ValueError:
            messagebox.showerror("Invalid Input", "Enter positive numbers for fabric count and dimensions.")
            return

        colours = self.colour_slider.get()
        contrast = self.contrast_slider.get()
        sharpen = self.sharpen_var.get()

        processed, labels, palette = quantize_image(
            self.original_image, width, height, colours, contrast=contrast, sharpen=sharpen
        )

        self.latest_quantized = processed
        self.latest_labels = labels
        self.latest_palette = palette

        self.show_image(processed.copy(), title="Processed Preview", row=1)

    def export_chart(self):
        if hasattr(self, 'latest_quantized') and hasattr(self, 'latest_labels') and hasattr(self, 'latest_palette'):
            save_chart_with_legend(self.latest_quantized, self.latest_labels, self.latest_palette)
        else:
            messagebox.showerror("Error", "Please update the preview first before exporting.")


if __name__ == "__main__":
    main_root = tk.Tk()
    app = CrossStitchApp(main_root)
    main_root.mainloop()
