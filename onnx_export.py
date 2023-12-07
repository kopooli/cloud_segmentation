import onnx
import torch
from model import CloudSegmenter
from tiling import TILE_SIZE


model = CloudSegmenter.load_from_checkpoint(
    "./lightning_logs/0,0001_minimal_cross_double_tiling_64_batch/checkpoints/checkpoints_train/epoch_epoch=18-step_step=7866.ckpt",
    arch="Linknet",
    encoder_name="timm-mobilenetv3_small_minimal_100",
    print_pictures=False,
)
model.eval()
batch_size = 32
input_tensor = torch.randn(batch_size, 4, TILE_SIZE, TILE_SIZE)
output = model(input_tensor)
assert list(output.shape) == [batch_size, TILE_SIZE, TILE_SIZE]
torch.onnx.export(
    model,
    input_tensor,
    "cloud_segmenter.onnx",
    export_params=True,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)

onnx_model = onnx.load("cloud_segmenter.onnx")
onnx.checker.check_model(onnx_model)
