import numpy as np
import pandas as pd
from load_scenes_by_categories import load_scenes_by_categories
from dataset import CloudDataset
from torchvision import transforms
from model import CloudSegmenter
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning as pl
import os
from torch.utils.data import DataLoader


train_scenes, validation_scenes, test_scenes = load_scenes_by_categories()
data_path = "./data/subscenes"
mask_path = "./data/masks"
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)
train_dataset = CloudDataset(data_path, mask_path, train_scenes, "train", transform)
test_dataset = CloudDataset(data_path, mask_path, test_scenes, "test", transform)
validation_dataset = CloudDataset(
    data_path, mask_path, validation_scenes, "validation", transform
)
n_cpu = os.cpu_count()
train_dataloader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=n_cpu
)
valid_dataloader = DataLoader(
    validation_dataset, batch_size=32, shuffle=False, num_workers=n_cpu
)
test_dataloader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=n_cpu
)

model = CloudSegmenter("Linknet", "timm-mobilenetv3_small_minimal_100")

callbacks = [
    ModelCheckpoint(
        save_last=True,
        save_top_k=-1,  # -1 keeps all, # << 0 keeps only last ....
        filename="checkpoints_train/epoch_{epoch:02d}-step_{step}",
    )
]

trainer = pl.Trainer(
    accelerator="cpu",
    max_epochs=20,
    callbacks=callbacks,
)

trainer.fit(
    model,
    train_dataloaders=train_dataloader,
    val_dataloaders=valid_dataloader,
)
