import os
import numpy as np
# import cv2 # Removed
import glob
import torch
import torchvision.transforms.functional as TF
import random
from PIL import Image # Keep Pillow import

class DataAugmentor:
    def __init__(self,
                 source_directory,
                 target_directory,
                 image_file_extension=".png",
                 flip_prob=0.5,  # Probability of applying horizontal or vertical flip
                 noise_prob=0.3,  # Probability of adding noise
                 noise_level=(0.02, 0.1),  # Range of noise intensity
                 contrast_prob=0.4,  # Probability of adjusting contrast
                 contrast_range=(0.7, 1.3)  # Range for contrast adjustment
                 ):

        print("Starting Data Augmentor")
        self.source_directory = source_directory
        self.target_directory = target_directory
        self.image_file_extension = image_file_extension

        self.flip_prob = flip_prob
        self.noise_prob = noise_prob
        self.noise_level = noise_level
        self.contrast_prob = contrast_prob
        self.contrast_range = contrast_range

        self.images = {}
        self.augmented_images = {}

        print(f"Looking for images in subdirectories of: {self.source_directory}")
        # Use os.path.join for robust path construction
        file_paths = glob.glob(os.path.join(self.source_directory, '*', '*' + self.image_file_extension))
        print(f"Found {len(file_paths)} images with pattern '{os.path.join(self.source_directory, '*', '*' + self.image_file_extension)}'")

        if not file_paths:
            print(f"No images found in subdirectories. Trying top-level directory: {self.source_directory}")
            # Adjusted glob pattern to include all files if imageFileExtension is general (e.g. ".*")
            # or specific files if imageFileExtension is specific (e.g. ".png")
            top_level_pattern = '*'
            if self.image_file_extension and self.image_file_extension != ".*":
                 top_level_pattern += self.image_file_extension

            file_paths = glob.glob(os.path.join(self.source_directory, top_level_pattern))
            print(f"Found {len(file_paths)} files with pattern '{os.path.join(self.source_directory, top_level_pattern)}'")
            if not file_paths and top_level_pattern != '*': # If specific extension yielded no results, try all files
                print(f"Still no files found with extension '{self.image_file_extension}'. Trying all files in top-level directory.")
                file_paths = glob.glob(os.path.join(self.source_directory, '*'))
                print(f"Found {len(file_paths)} files with pattern '{os.path.join(self.source_directory, '*')}'")


        loaded_image_count = 0
        for file_path in file_paths:
            try:
                # Directly open the image with Pillow. It loads in RGB by default.
                pil_img = Image.open(file_path)
                # Ensure image is in RGB mode if it has an alpha channel or is grayscale for consistency
                if pil_img.mode == 'RGBA' or pil_img.mode == 'LA':
                    pil_img = pil_img.convert('RGB')
                elif pil_img.mode == 'L': # Grayscale
                    pil_img = pil_img.convert('RGB') # Or handle as grayscale if your pipeline supports it

                self.images[file_path] = pil_img
                loaded_image_count +=1
            except FileNotFoundError:
                print(f"Error: File not found at {file_path}")
            except IOError: # Catches more general image opening errors
                print(f"Error: Could not open or read image file at {file_path}. It might be corrupted or not a supported format.")
            except Exception as e:
                print(f"An unexpected error occurred while loading {file_path}: {e}")

        print(f"Successfully loaded {loaded_image_count} images.")
        if not self.images:
            print("Warning: No images were loaded. Please check the source directory and file extensions.")


    def horizontal_flip(self, image: Image.Image) -> Image.Image:
        return TF.hflip(image)

    def vertical_flip(self, image: Image.Image) -> Image.Image:
        return TF.vflip(image)

    def add_noise(self, image: Image.Image) -> Image.Image:
        tensor_img = TF.to_tensor(image) # Converts PIL image to [C, H, W] tensor in [0.0, 1.0]
        noise_val = random.uniform(*self.noise_level)
        noise = torch.randn_like(tensor_img) * noise_val
        noisy_img_tensor = tensor_img + noise
        noisy_img_tensor = torch.clamp(noisy_img_tensor, 0, 1)
        return TF.to_pil_image(noisy_img_tensor) # Converts back to PIL Image

    def adjust_contrast(self, image: Image.Image) -> Image.Image:
        contrast_factor = random.uniform(*self.contrast_range)
        return TF.adjust_contrast(image, contrast_factor)

    def apply_augmentations(self, num_augmentations_per_image=4):
        if not self.images:
            print("No images loaded to augment. Skipping augmentation.")
            return

        print(f"Applying {num_augmentations_per_image} augmentations per image...")
        for path, img in self.images.items():
            augmented_list = []
            for _ in range(num_augmentations_per_image):
                aug_img = img.copy() # Start with a fresh copy for each augmentation sequence

                if random.random() < self.flip_prob:
                    aug_img = self.horizontal_flip(aug_img)
                if random.random() < self.flip_prob: # Independent probability for vertical flip
                    aug_img = self.vertical_flip(aug_img)
                if random.random() < self.noise_prob:
                    aug_img = self.add_noise(aug_img)
                if random.random() < self.contrast_prob:
                    aug_img = self.adjust_contrast(aug_img)

                augmented_list.append(aug_img)
            self.augmented_images[path] = augmented_list
        print(f"Finished applying augmentations. {len(self.augmented_images)} images have augmented versions.")


    def save_augmented_images(self):
        if not self.augmented_images:
            print("No augmented images to save.")
            return

        if not os.path.exists(self.target_directory):
            os.makedirs(self.target_directory)
            print(f"Created target directory: {self.target_directory}")

        total_saved_count = 0
        for original_path, aug_list in self.augmented_images.items():
            filename = os.path.basename(original_path)
            name_without_ext, original_ext = os.path.splitext(filename)

            # Use the original extension unless a specific one is forced by imageFileExtension
            save_extension = self.image_file_extension if self.image_file_extension else original_ext
            if not save_extension.startswith('.'): # ensure dot prefix
                save_extension = '.' + save_extension


            for i, aug_img in enumerate(aug_list):
                try:
                    # aug_img is already a PIL Image object
                    aug_filename = f"{name_without_ext}_aug{i + 1}{save_extension}"
                    save_path = os.path.join(self.target_directory, aug_filename)
                    aug_img.save(save_path)
                    total_saved_count += 1
                except Exception as e:
                    print(f"Error saving augmented image {aug_filename} to {save_path}: {e}")
        print(f"Saved {total_saved_count} augmented images to {self.target_directory}")