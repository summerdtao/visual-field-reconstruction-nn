import numpy as np
import torch
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    def __init__(self, data, label, normalize=False, transform=None):
        self.data = data
        self.label = label
        if normalize:
            self.data = normalize_data(data)
            self.label = [_ / 255 for _ in self.label]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.data[idx]
        label = self.label[idx]

        if self.transform:
            data = self.transform(data)

        sample = (data, label)

        return sample


# Normalize data column wise
def normalize_data(data):
    new_data = np.asarray(data)
    for i in range(data.shape[1]):
        new_data[:,i] = data[:,i] - min(data[:,i]) / max(data[:,i]) - min(data[:,i])
    return new_data.tolist()

