import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
import pytorch_lightning as pl
import torch
import segmentation_models_pytorch as smp
from tiling import get_masks_from_tiles, get_picture_from_tile



class CloudSegmenter(pl.LightningModule):
    def __init__(self, arch, encoder_name, print_pictures, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=4, classes=1, **kwargs
        )
        self.print_pictures = print_pictures
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        # self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, image):
        mask = self.model(image)
        if self.training:
            return mask
        prob_mask = mask.sigmoid()
        pred_mask = (prob_mask > 0.5).int()
        pred_mask = pred_mask.squeeze()
        return pred_mask

    def shared_evaluation_step_beginning(self, batch):
        images, masks = batch
        pred_masks = self.forward(images)
        masks = masks.squeeze()
        return pred_masks, masks, images

    def shared_evaluation_step_end(self, predicted_mask, truth_mask):
        tp, fp, fn, tn = smp.metrics.get_stats(
            predicted_mask.long(), truth_mask.long(), mode="binary"
        )
        self.tp += torch.sum(tp)
        self.fp += torch.sum(fp)
        self.fn += torch.sum(fn)
        self.tn += torch.sum(tn)

    def validation_step(self, batch, batch_idx):
        predicted_masks, truth_masks, _ = self.shared_evaluation_step_beginning(batch)
        self.shared_evaluation_step_end(predicted_masks, truth_masks)
        return None

    def on_validation_epoch_start(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        return

    def on_test_epoch_start(self) -> None:
        self.on_validation_epoch_start()
        self.id = 0
        return

    def on_validation_epoch_end(self):
        return self.shared_epoch_end("valid")

    def on_test_epoch_end(self) -> None:
        return self.shared_epoch_end(("testing"))

    def shared_epoch_end(self, stage):
        producer_accuracy = self.tp / (self.tp + self.fn)
        user_accuracy = self.tp / (self.tp + self.fp)
        balanced_accuracy = 0.5 * (producer_accuracy + (self.tn / (self.tn + self.fp)))
        metrics = {
            "producer_accuracy": producer_accuracy,
            "user_accuracy": user_accuracy,
            "overall_accuracy": balanced_accuracy,
        }
        self.log_dict(metrics, prog_bar=True)
        print(metrics)
        return

    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits_mask = self.forward(images)
        loss = self.loss_fn(logits_mask, masks.float())
        return loss

    def test_step(self, batch, batch_idx):
        predicted_masks, truth_masks, tiles = self.shared_evaluation_step_beginning(batch)
        pred_mask, masks = get_masks_from_tiles(predicted_masks, truth_masks.int())
        if self.print_pictures:
            self.save_pictures(pred_mask, masks, tiles)
        self.shared_evaluation_step_end(predicted_masks, truth_masks)
        return None

    def save_pictures(self, predicted_mask, truth_mask, image_tiles):
        pred_np, mask_np = predicted_mask.numpy(), truth_mask.numpy()
        whole_image = get_picture_from_tile(image_tiles)
        whole_image = whole_image[[0, 1, 2], :, :].numpy().transpose((1, 2, 0))
        whole_image = np.clip(whole_image, 0, 1) * 255

        path = "./save_images"
        if not os.path.exists(path):
            os.makedirs(path)
        plt.imsave(f"{path}/{self.id}_pred.png", pred_np, cmap=cm.gray, vmin=0, vmax=1)
        plt.imsave(f"{path}/{self.id}_truth.png", mask_np, cmap=cm.gray, vmin=0, vmax=1)
        plt.imsave(
            f"{path}/{self.id}_image.png",
            np.ascontiguousarray(whole_image.astype("uint8")),
        )
        self.id += 1

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)
