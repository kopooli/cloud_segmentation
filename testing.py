import os
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader
from model import CloudSegmenter
from tiling import TILES_PER_IMAGE
from load_scenes_by_categories import load_scenes_by_categories
from dataset import CloudDataset

train_scenes, validation_scenes, test_scenes = load_scenes_by_categories()
data_path = "./data/subscenes"
mask_path = "./data/masks"
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)
test_dataset = CloudDataset(data_path, mask_path, test_scenes, "test", transform)
validation_dataset = CloudDataset(
    data_path, mask_path, validation_scenes, "valid", transform
)
train_dataset = CloudDataset(data_path, mask_path, train_scenes, "train", transform)
n_cpu = os.cpu_count()
test_dataloader = DataLoader(
    test_dataset, batch_size=TILES_PER_IMAGE, shuffle=False, num_workers=1
)
validation_dataloader = DataLoader(
    validation_dataset, batch_size=TILES_PER_IMAGE, shuffle=False, num_workers=n_cpu
)
train_dataloader = DataLoader(
    train_dataset, batch_size=TILES_PER_IMAGE, shuffle=False, num_workers=n_cpu
)
#//home/jonas/cloud_segmentation/lightning_logs/0,0001_cross_entropy_smallest_normal_tiling_goood_split/checkpoints/checkpoints_train
model = CloudSegmenter.load_from_checkpoint(
    "./lightning_logs/0,0001_cross_entropy_smallest_normal_tiling_goood_split/checkpoints/checkpoints_train/epoch_epoch=21-step_step=5632.ckpt",
    arch="Linknet",
    encoder_name="timm-mobilenetv3_small_minimal_100",
    print_pictures=True,
)

trainer = pl.Trainer(
    accelerator="cpu",
    max_epochs=15,
)
# test_metrics = trainer.test(model, dataloaders=train_dataloader, verbose=False)
# test_metrics = trainer.test(model, dataloaders=validation_dataloader, verbose=False)
test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=False)
