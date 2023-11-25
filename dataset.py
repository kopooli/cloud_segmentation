from torch.utils.data import Dataset
import numpy as np
import os
from utils import get_tile
from torchvision import transforms

TILE_SIZE = 224
IMG_SIZE = 1022
# tiles does not fit perfectly into the image, so there is going to be overlap on 2 sides and one extra tile in corner

TILES_PER_IMAGE = (
    pow(TILE_SIZE*(IMG_SIZE // TILE_SIZE), 2) // pow(TILE_SIZE, 2)
    + 2 * (IMG_SIZE // TILE_SIZE)
    + 1
)
TILE_PER_DIMENSION = IMG_SIZE // TILE_SIZE + 1


class CloudDataset(Dataset):
    def __init__(self, file_path, mask_path, scenes, type, transform=None):
        self.file_path = file_path
        self.mask_path = mask_path
        self.scenes = scenes
        self.transform = transform
        self.type = type
        print(len(self.scenes) * TILES_PER_IMAGE)

    def __len__(self):
        return len(self.scenes)*TILES_PER_IMAGE

    def __getitem__(self, index):
        image_index = index // TILES_PER_IMAGE
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
