import os
import numpy as np
import cv2
import glob
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
from PIL import Image


class DataAugmentor:


    """
    Class for performing data augmentation on underwater images.
    Integrates with the existing DataManager and ImageManipulator classes.
    Implements horizontal/vertical flipping, noise addition, and contrast variation.
    """

    def __init__(self,
                 sourceDirectory,
                 targetDirectory,
                 imageFileExtension=".png",
                 flip_prob=0.5,  # Probability of applying horizontal or vertical flip
                 noise_prob=0.3,  # Probability of adding noise
                 noise_level=(0.02, 0.1),  # Range of noise intensity
                 contrast_prob=0.4,  # Probability of adjusting contrast
                 contrast_range=(0.7, 1.3)  # Range for contrast adjustment
                 ):
        """
        Initialize the DataAugmentor with source and target directories and augmentation parameters.

        Args:
            sourceDirectory: Directory containing the source images
            targetDirectory: Directory where augmented images will be saved
            imageFileExtension: File extension of images to augment
            flip_prob: Probability of applying horizontal/vertical flip
            noise_prob: Probability of adding noise
            noise_level: Range of noise intensity to add (min, max)
            contrast_prob: Probability of adjusting contrast
            contrast_range: Range for contrast adjustment (min, max)
        """
        print("Starting Data Augmentor")
        self.sourceDirectory = sourceDirectory
        self.targetDirectory = targetDirectory
        self.imageFileExtension = imageFileExtension

        # Augmentation parameters
        self.flip_prob = flip_prob
        self.noise_prob = noise_prob
        self.noise_level = noise_level
        self.contrast_prob = contrast_prob
        self.contrast_range = contrast_range

        # Dictionary to store original images (path: image)
        self.images = {}
        # Dictionary to store augmented versions of each image (path: list of augmented images)
        self.augmented_images = {}






        # Load all images from the source directory
        print(f"Looking for images in: {self.sourceDirectory}")
        filePaths = glob.glob(self.sourceDirectory + '/*/*' + self.imageFileExtension)
        print(f"Found {len(filePaths)} images")

        # If you found no images, try a different pattern
        if len(filePaths) == 0:
            print("Trying different pattern...")
            filePaths = glob.glob(self.sourceDirectory + '/*')
            print(f"Found {len(filePaths)} files with any extension")
            # Check what extensions are present
            extensions = set(os.path.splitext(f)[1] for f in filePaths)
            print(f"Extensions found: {extensions}")


        for filePath in filePaths:
            # Load image using OpenCV (which uses BGR color format)
            cv_img = cv2.imread(filePath)
            # Convert to RGB for compatibility with torchvision (which uses RGB)
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image for compatibility with torchvision transforms
            pil_img = Image.fromarray(rgb_img)
            self.images[filePath] = pil_img

    def horizontal_flip(self, image):
        """Apply horizontal flip to an image."""
        return TF.hflip(image)

    def vertical_flip(self, image):
        """Apply vertical flip to an image."""
        return TF.vflip(image)

    def add_noise(self, image):
        """Add random Gaussian noise to an image."""
        # Convert PIL image to tensor (scales values to [0,1])
        tensor_img = TF.to_tensor(image)

        # Generate random noise intensity from the specified range
        noise_level = random.uniform(*self.noise_level)

        # Create noise tensor with same shape as image
        noise = torch.randn_like(tensor_img) * noise_level

        # Add noise to image
        noisy_img = tensor_img + noise

        # Clip values to valid range [0, 1]
        noisy_img = torch.clamp(noisy_img, 0, 1)

        # Convert back to PIL image
        return TF.to_pil_image(noisy_img)

    def adjust_contrast(self, image):
        """Adjust the contrast of an image."""
        # Generate random contrast factor from the specified range
        contrast_factor = random.uniform(*self.contrast_range)

        # Apply contrast adjustment
        return TF.adjust_contrast(image, contrast_factor)

    def apply_augmentations(self, num_augmentations=4):
        """
        Apply random augmentations to each image.

        Args:
            num_augmentations: Number of augmented versions to create for each image
        """
        for path, img in self.images.items():
            # Initialize list to store augmented versions of this image
            augmented_list = []

            # Generate specified number of augmented images
            for _ in range(num_augmentations):
                # Start with a copy of the original image
                aug_img = img.copy()

                # Apply horizontal flip with probability
                if random.random() < self.flip_prob:
                    aug_img = self.horizontal_flip(aug_img)

                # Apply vertical flip with probability
                if random.random() < self.flip_prob:
                    aug_img = self.vertical_flip(aug_img)

                # Apply noise with probability
                if random.random() < self.noise_prob:
                    aug_img = self.add_noise(aug_img)

                # Apply contrast adjustment with probability
                if random.random() < self.contrast_prob:
                    aug_img = self.adjust_contrast(aug_img)

                # Add the augmented image to the list
                augmented_list.append(aug_img)

            # Store all augmented versions of this image
            self.augmented_images[path] = augmented_list

    def save_augmented_images(self):
        """Save all augmented images to the target directory."""
        # Create target directory if it doesn't exist
        if not os.path.exists(self.targetDirectory):
            os.makedirs(self.targetDirectory)

        # Save each augmented image
        for original_path, aug_list in self.augmented_images.items():
            # Get the filename without extension
            filename = os.path.basename(original_path)
            name_without_ext = os.path.splitext(filename)[0]

            # Save each augmented version with a suffix
            for i, aug_img in enumerate(aug_list):
                # Convert PIL image back to OpenCV format for saving
                opencv_img = cv2.cvtColor(np.array(aug_img), cv2.COLOR_RGB2BGR)

                # Create augmented filename
                aug_filename = f"{name_without_ext}_aug{i + 1}{self.imageFileExtension}"
                save_path = os.path.join(self.targetDirectory, aug_filename)

                # Save the image
                cv2.imwrite(save_path, opencv_img)

        print(f"Saved {sum(len(v) for v in self.augmented_images.values())} augmented images to {self.targetDirectory}")

