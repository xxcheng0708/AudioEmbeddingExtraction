# coding:utf-8
"""
    Created by cheng star at 2022/1/23 13:03
    @email : xxcheng0708@163.com
"""
from torchvision.io import read_image
import numpy as np
from torch.utils.data import Dataset


class AudioNpyDataset(Dataset):
    def __init__(self, filenames, labels, transforms, down_sample=False):
        self.filenames = filenames
        self.labels = labels
        self.transforms = transforms
        self.down_sample = down_sample

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        data = np.load(self.filenames[idx]).astype(np.float32)
        if self.down_sample:
            data = data[:, ::4]
        data = self.transforms(data)
        # print(data.shape)
        return data, self.labels[idx]


class ImageDataset(Dataset):
    def __init__(self, filenames, labels, transforms):
        self.filenames = filenames
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = read_image(self.filenames[idx])
        image = self.transforms(image)
        return image, self.labels[idx]


