import numpy as np
import torch
from torch.utils.data import Dataset

class MultispectralDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        Args:
            images (list or np.ndarray): List of input images.
            labels (list or np.ndarray): Corresponding labels for the images.
            transform (callable, optional): Optional transform to be applied to the images.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # # Apply the transform if specified
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)
