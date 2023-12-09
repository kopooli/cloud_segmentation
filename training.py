import pytorch_lightning as pl
import os
from torchvision import transforms
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.utils.data import DataLoader

from model import CloudSegmenter
from load_scenes_by_categories import load_scenes_by_categories
from dataset import CloudDatasetDoubleTiles, CloudDataset


train_scenes, validation_scenes, test_scenes = load_scenes_by_categories()
data_path = "./data/subscenes"
mask_path = "./data/masks"
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)
train_dataset = CloudDatasetDoubleTiles(data_path, mask_path, train_scenes, transform)
test_dataset = CloudDataset(data_path, mask_path, test_scenes, transform)
validation_dataset = CloudDataset(data_path, mask_path, validation_scenes, transform)
n_cpu = os.cpu_count()
train_dataloader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=n_cpu - 1
)
valid_dataloader = DataLoader(
    validation_dataset, batch_size=32, shuffle=False, num_workers=n_cpu - 1
)
test_dataloader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=n_cpu - 1
)

model = CloudSegmenter(
    "Linknet", "timm-mobilenetv3_small_minimal_100", print_pictures=False
)

callbacks = [
    ModelCheckpoint(
        save_last=True,
        save_top_k=-1,  # -1 keeps all, # << 0 keeps only last ....
        filename="checkpoints_train/epoch_{epoch:02d}-step_{step}",
    )
]

trainer = pl.Trainer(
    accelerator="cpu",
    max_epochs=30,
    callbacks=callbacks,
)

trainer.fit(
    model,
    train_dataloaders=train_dataloader,
    val_dataloaders=valid_dataloader,
)
