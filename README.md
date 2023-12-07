# Cloud segmentation
This repository contains my cloud segmentation project. It is trained using [this dataset](https://zenodo.org/records/4172871). All scripts suppose the images are in ./data/subscenes and corresponding ground truth masks in ./data/masks, except for onnx_inference.py, which accepts your paths to files as arguments. But if you do not pass it, it will look for it in the ./data/ subfolder as other scripts. All scripts are using cpu.
## ONNX
My best model is exported as "cloud_segmenter.onnx" in this folder. You can run inference on it using "onnx_inference.py". Simply run 
```bash
python onnx_inference.py path/to/your/images path/to/your/masks
```
The images and masks must have 1022x1022 resolution. Each image and corresponding mask must also have the same name. Images must be in Sentinel 2 L1A HxWxC format (each image must have a shape (1022, 1022, 13)) and saved as NumPy array (.npy).

The inference collects a sum of true positives, false positives, false negatives and true negatives. Ultimately, it computes the metrics from them (Producer accuracy, User accuracy and Balanced overall accuracy) and prints them. You can understand it as (Recall, Precision, Balanced score of recall and precision).

Jonáš Herec
