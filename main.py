import numpy as np
import pandas as pd
from load_scenes_by_categories import load_scenes_by_categories
from dataset import CloudDataset
from torchvision import transforms

train_scenes, validation_scenes, test_scenes = load_scenes_by_categories()
data_path = "./data/subscenes"
mask_path = "./data/masks"
transform = transforms.Compose([
    transforms.ToTensor(),
])
train_dataset = CloudDataset(data_path, mask_path, train_scenes, transform)
test_dataset = CloudDataset(data_path, mask_path, test_scenes, transform)
validation_dataset = CloudDataset(data_path, mask_path, validation_scenes, transform)
sample = train_dataset[0]
print(sample)