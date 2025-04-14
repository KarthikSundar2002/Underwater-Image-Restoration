from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms

# PyTorch Dataset for UIEB
class UIEBDataset(Dataset):
    def __init__(self, raw_dir, ref_dir, transform=None):

        self.raw_images = sorted(os.listdir(raw_dir))
        self.ref_images = sorted(os.listdir(ref_dir))
        self.raw_dir = raw_dir
        self.ref_dir = ref_dir
        self.transform = transform

        if self.transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.raw_images)

    def __getitem__(self, idx):
        raw_path = os.path.join(self.raw_dir, self.raw_images[idx])
        ref_path = os.path.join(self.ref_dir, self.ref_images[idx])

        raw_img = Image.open(raw_path).convert("RGB")
        ref_img = Image.open(ref_path).convert("RGB")

        if self.transform:
            raw_img = self.transform(raw_img)
            ref_img = self.transform(ref_img)

        return raw_img, ref_img  # Input and Ground Truth
