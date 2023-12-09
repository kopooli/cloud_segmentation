import onnxruntime
import os
import segmentation_models_pytorch as smp
import sys
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import CloudDataset
from tiling import TILES_PER_IMAGE, get_masks_from_tiles


tp_sum = 0
fp_sum = 0
fn_sum = 0
tn_sum = 0
if len(sys.argv) not in [1, 3]:
    raise Exception(
        "Invalid number of arguments passed, either pass path/to/images path/to/labels or do not pass anything."
    )
images_dir = "./data/subscenes/"
label_masks_dir = "./data/masks/"
if len(sys.argv) == 3:
    images_dir = sys.argv[1]
    label_masks_dir = sys.argv[2]

if os.listdir(images_dir) != os.listdir(label_masks_dir):
    raise Exception(
        "Labels does not match input images. There must be same number of them and they must have same names."
    )
batch_size = TILES_PER_IMAGE
save_images = True
ort_session = onnxruntime.InferenceSession(
    "cloud_segmenter.onnx", providers=["CPUExecutionProvider"]
)
image_names = [image_name.replace(".npy", "") for image_name in os.listdir(images_dir)]
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)
inference_dataset = CloudDataset(images_dir, label_masks_dir, image_names, transform)
n_cpu = os.cpu_count()
inference_dataloader = DataLoader(
    inference_dataset, batch_size=batch_size, shuffle=False, num_workers=1
)
for batch in tqdm(inference_dataloader, "Evaluating by picture"):
    image, mask = batch
    mask = mask.squeeze()
    ort_inputs = {"input": image.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    pred_mask, truth_mask = get_masks_from_tiles(torch.from_numpy(ort_outs[0]), mask)
    tp, fp, fn, tn = smp.metrics.get_stats(
        truth_mask.long(), pred_mask.long(), mode="binary"
    )
    tp_sum += torch.sum(tp)
    fp_sum += torch.sum(fp)
    fn_sum += torch.sum(fn)
    tn_sum += torch.sum(tn)
producer_accuracy = tp_sum / (tp_sum + fn_sum)
user_accuracy = tp_sum / (tp_sum + fp_sum)
balanced_accuracy = 0.5 * (producer_accuracy + (tn_sum / (tn_sum + fp_sum)))
final_string = f"FINAL RESULTS\n#############\nProducer accuracy: {producer_accuracy}\nUser accuracy: {user_accuracy}\nOverall accuracy {balanced_accuracy}\n#############\n"
print(final_string)
