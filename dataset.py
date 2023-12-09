import numpy as np
import os
import tiling
from torch.utils.data import Dataset


class CloudDataset(Dataset):
    def __init__(self, file_path, mask_path, scenes, transform=None):
        self.file_path = file_path
        self.mask_path = mask_path
        self.scenes = scenes
        self.transform = transform

    def __len__(self):
        return len(self.scenes) * tiling.TILES_PER_IMAGE

    def __getitem__(self, index):
        image_index = index // tiling.TILES_PER_IMAGE
        image, mask = self.shared_get_item_logic(index, image_index, double=False)
        return image, mask

    def shared_get_item_logic(self, index, image_index, double):
        desired_channels = [3, 2, 1, 7]
        cloudy_channel = 1
        scene = self.scenes[image_index]
        image = np.load(os.path.join(self.file_path, f"{scene}.npy"))[
            :, :, desired_channels
        ]
        mask = np.load(os.path.join(self.mask_path, f"{scene}.npy"))[
            :, :, cloudy_channel
        ]
        image, mask = tiling.get_tile(index, image, mask, double=double)
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask).int()
        return image, mask


class CloudDatasetDoubleTiles(CloudDataset):
    def __len__(self):
        return len(self.scenes) * tiling.DOUBLE_TILES_PER_IMAGE

    def __getitem__(self, index):
        image_index = index // tiling.DOUBLE_TILES_PER_IMAGE
        image, mask = self.shared_get_item_logic(index, image_index, double=True)
        return image, mask
