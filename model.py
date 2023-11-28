import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from tiling import get_picture_from_tiles


class CloudSegmenter(pl.LightningModule):
    def __init__(self, arch, encoder_name, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=4, classes=1, **kwargs
        )

        # for image segmentation dice loss could be the best first choice
        #self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, image):
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        images, masks = batch
        logits_mask = self.forward(images)
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, masks.float())
        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), masks.long(), mode="binary"
        )
        self.tp += torch.sum(tp)
        self.fp += torch.sum(fp)
        self.fn += torch.sum(fn)
        self.tn += torch.sum(tn)
        return loss

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        return

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        return

    def on_validation_epoch_end(self):
        return self.shared_epoch_end("valid")

    def on_test_epoch_end(self) -> None:
        return self.shared_epoch_end(("testing"))

    def shared_epoch_end(self, stage):
        # aggregate step metics
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

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def test_step(self, batch, batch_idx):
        images, masks = batch
        logits_mask = self.forward(images)
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, masks.float())
        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).int()
        pred_mask, masks = get_picture_from_tiles(pred_mask, masks)
        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), masks.long(), mode="binary"
        )
        self.tp += torch.sum(tp)
        self.fp += torch.sum(fp)
        self.fn += torch.sum(fn)
        self.tn += torch.sum(tn)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)
