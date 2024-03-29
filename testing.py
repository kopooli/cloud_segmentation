import os
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import CloudDataset
from load_scenes_by_categories import load_scenes_by_categories
from model import CloudSegmenter
from tiling import TILES_PER_IMAGE


train_scenes, validation_scenes, test_scenes = load_scenes_by_categories()
data_path = "./data/subscenes"
mask_path = "./data/masks"
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)
test_dataset = CloudDataset(data_path, mask_path, test_scenes, transform)
validation_dataset = CloudDataset(data_path, mask_path, validation_scenes, transform)
train_dataset = CloudDataset(data_path, mask_path, train_scenes, transform)
n_cpu = os.cpu_count()
test_dataloader = DataLoader(
    test_dataset, batch_size=TILES_PER_IMAGE, shuffle=False, num_workers=n_cpu
)
validation_dataloader = DataLoader(
    validation_dataset, batch_size=TILES_PER_IMAGE, shuffle=False, num_workers=n_cpu
)
train_dataloader = DataLoader(
    train_dataset, batch_size=TILES_PER_IMAGE, shuffle=False, num_workers=n_cpu
)
model = CloudSegmenter.load_from_checkpoint(
    "./lightning_logs/0,0001_minimal_cross_double_tiling_64_batch/checkpoints/checkpoints_train/epoch_epoch=18-step_step=7866.ckpt",
    arch="Linknet",
    encoder_name="timm-mobilenetv3_small_minimal_100",
    print_pictures=True,
)

trainer = pl.Trainer(
    accelerator="cpu",
    max_epochs=15,
)
test_metrics = trainer.test(model, dataloaders=train_dataloader, verbose=False)
test_metrics = trainer.test(model, dataloaders=validation_dataloader, verbose=False)
test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=False)
