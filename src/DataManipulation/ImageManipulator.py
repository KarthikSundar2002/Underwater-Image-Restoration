import os
import numpy as np
from PIL import Image # Import the Image module from Pillow
import glob

# Will hold the images in memory, use with a WITH statement to avoid memory issues
# for this reason will not be most efficient on larger datasets
# better approach would be to queue operations
class ImageManipulator:
    def __init__(self, source_directory, target_directory, image_file_extension=".png"):
        self.source_directory = source_directory
        self.target_directory = target_directory
        self.image_file_extension = image_file_extension

        self.images = {}
        file_paths = glob.glob(os.path.join(self.source_directory, '*' + self.image_file_extension))
        for file_path in file_paths:
            try:
                # Use Pillow to open the image
                img = Image.open(file_path)
                # Keep the image as a Pillow Image object.
                # If you need to work with it as a NumPy array later for other reasons,
                # you can convert it then, but for reading/resizing/saving,
                # Pillow objects are fine.
                self.images.update({file_path: img})
            except Exception as e:
                print(f"Error opening image {file_path}: {e}")


    def resize_images(self, height=256, width=256):
        resized_images = {}
        for file_path, img in self.images.items():
            try:
                # Pillow's resize method.
                # Image.Resampling.BICUBIC is equivalent to cv2.INTER_CUBIC
                resized_img = img.resize((width, height), Image.Resampling.BICUBIC)
                resized_images.update({file_path: resized_img})
            except Exception as e:
                print(f"Error resizing image {file_path}: {e}")
        self.images = resized_images # Update the dictionary with resized images
        return

    def save_to_disk(self):
        if not os.path.exists(self.target_directory):
            os.makedirs(self.target_directory)

        for file_path, img in self.images.items():
            try:
                # Construct the full save path
                save_path = os.path.join(self.target_directory, os.path.basename(file_path))
                # Use Pillow's save method
                img.save(save_path)
            except Exception as e:
                print(f"Error saving image {save_path}: {e}")