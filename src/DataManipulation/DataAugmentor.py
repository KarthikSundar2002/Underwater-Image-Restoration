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

        print("Starting Data Augmentor")
        self.sourceDirectory = sourceDirectory
        self.targetDirectory = targetDirectory
        self.imageFileExtension = imageFileExtension

        self.flip_prob = flip_prob
        self.noise_prob = noise_prob
        self.noise_level = noise_level
        self.contrast_prob = contrast_prob
        self.contrast_range = contrast_range

        self.images = {}
        self.augmented_images = {}

        print(f"Looking for images in: {self.sourceDirectory}")
        filePaths = glob.glob(self.sourceDirectory + '/*/*' + self.imageFileExtension)
        print(f"Found {len(filePaths)} images")

        if len(filePaths) == 0:
            print("Trying different pattern...")
            filePaths = glob.glob(self.sourceDirectory + '/*')
            print(f"Found {len(filePaths)} files with any extension")
            extensions = set(os.path.splitext(f)[1] for f in filePaths)
            print(f"Extensions found: {extensions}")

        for filePath in filePaths:
            cv_img = cv2.imread(filePath)
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)
            self.images[filePath] = pil_img

    def horizontal_flip(self, image):
        return TF.hflip(image)

    def vertical_flip(self, image):
        return TF.vflip(image)

    def add_noise(self, image):
        tensor_img = TF.to_tensor(image)
        noise_level = random.uniform(*self.noise_level)
        noise = torch.randn_like(tensor_img) * noise_level
        noisy_img = tensor_img + noise
        noisy_img = torch.clamp(noisy_img, 0, 1)
        return TF.to_pil_image(noisy_img)

    def adjust_contrast(self, image):
        contrast_factor = random.uniform(*self.contrast_range)
        return TF.adjust_contrast(image, contrast_factor)

    def apply_augmentations(self, num_augmentations=4):
        for path, img in self.images.items():
            augmented_list = []
            for _ in range(num_augmentations):
                aug_img = img.copy()

                if random.random() < self.flip_prob:
                    aug_img = self.horizontal_flip(aug_img)
                if random.random() < self.flip_prob:
                    aug_img = self.vertical_flip(aug_img)
                if random.random() < self.noise_prob:
                    aug_img = self.add_noise(aug_img)
                if random.random() < self.contrast_prob:
                    aug_img = self.adjust_contrast(aug_img)

                augmented_list.append(aug_img)
            self.augmented_images[path] = augmented_list

    def save_augmented_images(self):
        if not os.path.exists(self.targetDirectory):
            os.makedirs(self.targetDirectory)

        for original_path, aug_list in self.augmented_images.items():
            filename = os.path.basename(original_path)
            name_without_ext = os.path.splitext(filename)[0]

            for i, aug_img in enumerate(aug_list):
                opencv_img = cv2.cvtColor(np.array(aug_img), cv2.COLOR_RGB2BGR)
                aug_filename = f"{name_without_ext}_aug{i + 1}{self.imageFileExtension}"
                save_path = os.path.join(self.targetDirectory, aug_filename)
                cv2.imwrite(save_path, opencv_img)
        print(f"Saved {sum(len(v) for v in self.augmented_images.values())} augmented images to {self.targetDirectory}")