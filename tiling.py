import numpy as np
import pandas as pd
from load_scenes_by_categories import load_scenes_by_categories
from dataset import CloudDataset
from torchvision import transforms

train_scenes, validation_scenes, test_scenes = load_scenes_by_categories()
