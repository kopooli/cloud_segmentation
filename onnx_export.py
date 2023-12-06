from model import CloudSegmenter
import torch
from tiling import TILE_SIZE
from torch import onnx

model = CloudSegmenter.load_from_checkpoint(
    "./lightning_logs/version_157/checkpoints/checkpoints_train/epoch_epoch=09-step_step=4140.ckpt",
    arch="Linknet",
    encoder_name="timm-mobilenetv3_small_minimal_100",
    print_pictures = False,
)
model.eval()
batch_size = 32
input_tensor = torch.randn(batch_size, 4, TILE_SIZE, TILE_SIZE)
output = model(input_tensor)
onnx.export(model, input_tensor)