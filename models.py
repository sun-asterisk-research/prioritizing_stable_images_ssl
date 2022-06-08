#  Import pytorch lightning
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch


class PrioritizingStableSegmentationModel(pl.LightningModule):

    def __init__(self, arch='', encoder_name='', in_channels=3, out_classes=1, checkpoint_path='',
                 labeled_dataloader=None, unlabeled_dataloader=None, is_semi=False,
                 teacher=None, use_soft_label=True, **kwargs):
        super().__init__()

        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        # max iou of origin model and momentum model
        self.m_max_iou = 0
        self.mm_max_iou = 0
        self.checkpoint_path = checkpoint_path

        self.labeled_dataloader = labeled_dataloader
        self.unlabeled_dataloader = unlabeled_dataloader
        self.teacher = teacher
        self.is_semi = is_semi
        self.use_soft_label = use_soft_label
        # Beta for unlabeled loss
        self.beta = 0.3

        # Non back-propagate into teacher
        if self.is_semi and self.teacher:
            for param in self.teacher.parameters():
                param.requires_grad = False

    def forward(self, image):
        return self.model(image)

    def make_pseudo_label(self, image):
        logits = self.teacher.cuda()(image)
        prob_mask = logits.sigmoid()
        # Generate hard label
        pred_mask = (prob_mask > 0.5).float()
        if self.use_soft_label:
            return prob_mask
        return pred_mask

    def compute_loss(self, image, mask):
        assert image.ndim == 4
        assert mask.ndim == 4

        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0
        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        result = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "loss": loss
        }
        return result

    def shared_step(self, batch, stage):
        # If not have unlabeled data, the batch is only have labeled data
        batch_labeled = batch

        if stage == 'train':
            batch_labeled = batch['labeled']

        # Batch of labeled
        l_image = batch_labeled["image"]
        l_mask = batch_labeled["mask"]

        l_result = self.compute_loss(l_image, l_mask)
        l_loss = l_result['loss']
        # Predefine for unlabeled loss
        u_result = l_result
        u_result['loss'] = 0
        u_loss = u_result['loss']

        # Batch of unlabeled
        if self.is_semi and stage == 'train':
            batch_unlabeled = batch['unlabeled']
            u_image = batch_unlabeled["image"]
            # Compute mask from unlabeled image with teacher model
            u_mask = self.make_pseudo_label(batch_unlabeled["image"])
            u_result = self.compute_loss(u_image, u_mask)
            u_loss = u_result['loss']

        # Compute total loss
        loss = l_loss + self.beta * u_loss

        self.log("{}/labeled_loss".format(stage), l_loss)

        self.log("{}/unlabeled_loss".format(stage), u_loss)

        self.log("{}/loss".format(stage), loss)

        # Append loss to result
        result = l_result
        result['loss'] = loss

        return result

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        per_image_dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")

        if stage == 'valid':
            # Save best checkpoint
            if per_image_iou > self.m_max_iou:
                print('\nSave best model with valid loss = {}'.format(per_image_iou))
                torch.save(self.model, self.checkpoint_path)
                # In training teacher save K checkpoints
                if self.is_semi is False:
                    print('\nSave model in current epoch with valid loss = {}'.format(per_image_iou))
                    torch.save(self.model, "{}.epoch_{}".format(self.checkpoint_path, self.current_epoch))
                    self.m_max_iou = per_image_iou

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_per_image_dice": per_image_dice,
        }

        self.log_dict(metrics, prog_bar=True)

    def train_dataloader(self):
        if self.is_semi:
            loaders = {"labeled": self.labeled_dataloader, "unlabeled": self.unlabeled_dataloader}
        else:
            loaders = {"labeled": self.labeled_dataloader}

        return loaders

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)
