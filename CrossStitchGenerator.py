# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 16:55:34 2025

@author: ellie
"""

from PIL import Image
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def get_image_path():
    root = Tk()
    root.withdraw()  # Hide the main window
    file_path = askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")]
    )
    return file_path    
    
if __name__ == "__main__":
    image_path = get_image_path()
    if image_path:
        print(f"Selected file: {image_path}")
    else:
        print("No file selected.")
    
img = Image.open(image_path)
img.show()  # just to verify it loaded correctly