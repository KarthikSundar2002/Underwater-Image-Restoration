import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import src.DataManipulation.UIEBDataset as UIEBDataset

# Function to get DataLoaders
def get_dataloaders(raw_dir, ref_dir, batch_size=16, num_workers=4):

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(),
        #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
    ])

    #todo: NOTE: Do we want to be running this again as well as augmenting images?
    dataset = UIEBDataset.UIEBDataset(raw_dir, ref_dir, transform=transform)
    train_size = int(0.8 * len(dataset))  # 80% train, 20% test
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
