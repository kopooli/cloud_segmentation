from torch.utils.data import Dataset
import numpy as np
import os
TILE_SIZE = 224
class CloudDataset(Dataset):
    def __init__(self, file_path, mask_path, scenes, transform=None):
        self.file_path = file_path
        self.mask_path = mask_path
        self.scenes = scenes
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        # Load image and mask
        #RGB + NIR
        desired_channels = [3,2,1,7]
        cloudy_channel = 1
        scene = self.scenes[index]
        image = np.load(os.path.join(self.file_path, f"{scene}.npy"))[:,:,desired_channels]
        mask = np.load(os.path.join(self.mask_path, f"{scene}.npy"))[:,:,cloudy_channel]

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask).int()

        return image, mask