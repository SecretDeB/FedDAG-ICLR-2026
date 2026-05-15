import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        Args:
            data (list of tuples): A list where each tuple is (image, label).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]

        if self.transform:
            # Convert image from numpy array to PIL image for compatibility with torchvision transforms
            image = transforms.functional.to_pil_image(image)

            image = self.transform(image)

        return image, label


def getCustomDataLd(data):
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) ])

    dataset = CustomDataset(data, transform=transform)
    data_loader = DataLoader(dataset=dataset, batch_size=6, shuffle=True)
    return data_loader

# Now train_loader and test_loader are ready to be used for model training and evaluation.
