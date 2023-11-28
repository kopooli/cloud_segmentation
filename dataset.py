from torch.utils.data import Dataset
import numpy as np
import os
from tiling import get_tile
from torchvision import transforms
import tiling



class CloudDataset(Dataset):
    def __init__(self, file_path, mask_path, scenes, type, transform=None):
        self.file_path = file_path
        self.mask_path = mask_path
        self.scenes = scenes
        self.transform = transform
        self.type = type
        print(len(self.scenes) * tiling.TILES_PER_IMAGE)

    def __len__(self):
        return len(self.scenes) * tiling.TILES_PER_IMAGE

    def __getitem__(self, index):
        image_index = index // tiling.TILES_PER_IMAGE
        # RGB + NIR
        desired_channels = [3, 2, 1, 7]
        cloudy_channel = 1
        scene = self.scenes[image_index]
        image = np.load(os.path.join(self.file_path, f"{scene}.npy"))[
            :, :, desired_channels
        ]
        mask = np.load(os.path.join(self.mask_path, f"{scene}.npy"))[
            :, :, cloudy_channel
        ]
        image, mask = get_tile(index, image, mask)
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask).int()
        return image, mask
