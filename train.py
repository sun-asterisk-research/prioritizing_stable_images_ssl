import glob
import os


# Apply WanDB
import wandb
from datasets import PolypDataset
from hyperparams import (
    batch_size,
    output_dir,
    project,
    seed,
    device,
    confidence_threshold,
    test_size,
    trainsize,
    val_size,
    n_epochs,
    wandb_host,
    wandb_key
)
from seeder import set_seed_everything
from transforms import semi_transform, train_transform, val_transform

# Import model
from models import PrioritizingStableSegmentationModel
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils import generate_image_scores, pickle_dump, pickle_load, prioritizer

# Reconfig your API Key here
os.environ["WANDB_API_KEY"] = wandb_key
os.environ["WANDB_BASE_URL"] = wandb_host

# Create output dir
os.makedirs(output_dir, exist_ok=True)
# Login wandb
wandb.login()

set_seed_everything(seed)

f_images = glob.glob("data/TrainDataset/images/*")
f_masks = glob.glob("data/TrainDataset/masks/*")

X_train, X_val, y_train, y_val = train_test_split(
    f_images, f_masks, test_size=val_size, random_state=seed
)
# Use 40% for labeled
X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(
    X_train, y_train, test_size=test_size, random_state=seed
)

# Setup dataset
val_dataset = PolypDataset(
    X_val, y_val, trainsize=trainsize, transform=val_transform
)

labeled_dataset = PolypDataset(
    X_labeled, y_labeled, trainsize=trainsize, transform=train_transform
)

unlabeled_dataset = PolypDataset(
    X_unlabeled, y_unlabeled, trainsize=trainsize, transform=semi_transform
)

print(f"Valid size: {len(val_dataset)}")

n_cpu = os.cpu_count()

valid_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=n_cpu,
    pin_memory=True,
)

labeled_dataloader = DataLoader(
    labeled_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=n_cpu,
    pin_memory=True,
)


def train_model(model, experiment_name, max_epochs):
    wandb_logger = WandbLogger(project=project, name=experiment_name, log_model=False)
    trainer = pl.Trainer(gpus=1, logger=wandb_logger, max_epochs=max_epochs)
    trainer.fit(model, val_dataloaders=valid_dataloader)
    wandb.finish()


# Training teacher model
print("TRAINING TEACHER MODEL...")
model = PrioritizingStableSegmentationModel(
    "FPN",
    "densenet169",
    in_channels=3,
    out_classes=1,
    labeled_dataloader=labeled_dataloader,
    unlabeled_dataloader=None,
    checkpoint_path="{}/fcn_densenet169_teacher.pth".format(output_dir),
)
experiment_name = "Train teacher model - labeled ratio = {} %".format(
    100 - int(test_size * 100)
)

train_model(model, experiment_name, n_epochs)

# Calculate stable images to save image_mean_score.pkl.
# This function take along time (about 2 hours to compute with 30 checkpoints)
print("CALCULATE STABLE SCORE FOR ALL UNLABELED IMAGES...")
checkpoint_path = "{}/fcn_densenet169_teacher.pth".format(output_dir)
best_model = torch.load(checkpoint_path)
best_model.to(device)
checkpoint_paths = glob.glob("{}.epoch_*".format(checkpoint_path))
# Generate mean score dictionary for each image
mean_score_dict = generate_image_scores(best_model, checkpoint_paths, unlabeled_dataset)
# Dump to pickle
pickle_dump(mean_score_dict, 'image_mean_score.pkl')
# Load from pickle
mean_score_dict = pickle_load('image_mean_score.pkl')

# Get high/low confidence images
hc_images, lc_images = prioritizer(mean_score_dict, confidence_threshold)

# Re-define the unlabeled dataset but not need the masks
unlabeled_dataset = PolypDataset(
    hc_images, hc_images, trainsize=trainsize, transform=semi_transform
)

labeled_dataloader = DataLoader(
    labeled_dataset,
    batch_size=batch_size // 2,
    shuffle=True,
    num_workers=n_cpu,
    pin_memory=True,
)

unlabeled_dataloader = DataLoader(
    unlabeled_dataset,
    batch_size=batch_size // 2,
    shuffle=True,
    num_workers=n_cpu,
    pin_memory=True,
)

# Training teacher model
print("RE-TRAINING 1ST STUDENT WITH HIGH CONFIDENCE IMAGES...")
model = PrioritizingStableSegmentationModel(
    "FPN",
    "densenet169",
    in_channels=3,
    out_classes=1,
    is_semi=True,
    teacher=best_model,
    labeled_dataloader=labeled_dataloader,
    unlabeled_dataloader=unlabeled_dataloader,
    checkpoint_path="{}/fcn_densenet169_student_hc.pth".format(output_dir),
)
experiment_name = "Train 1st student with high confidence images - labeled ratio = {} %".format(
    100 - int(test_size * 100)
)

train_model(model, experiment_name, n_epochs)

# Re-define the unlabeled dataset but not need the masks. Use full dataset include high confidence and low confidence
unlabeled_dataset = PolypDataset(
    X_unlabeled, X_unlabeled, trainsize=trainsize, transform=semi_transform
)

labeled_dataloader = DataLoader(
    labeled_dataset,
    batch_size=batch_size // 2,
    shuffle=True,
    num_workers=n_cpu,
    pin_memory=True,
)

unlabeled_dataloader = DataLoader(
    unlabeled_dataset,
    batch_size=batch_size // 2,
    shuffle=True,
    num_workers=n_cpu,
    pin_memory=True,
)

checkpoint_path = "{}/fcn_densenet169_student_hc.pth".format(output_dir)
hc_student = torch.load(checkpoint_path)
hc_student.to(device)

# Training teacher model
print("RE-TRAINING 2ND STUDENT WITH HIGH CONFIDENCE AND LOW CONFIDENCE IMAGES...")
model = PrioritizingStableSegmentationModel(
    "FPN",
    "densenet169",
    in_channels=3,
    out_classes=1,
    is_semi=True,
    teacher=hc_student,
    labeled_dataloader=labeled_dataloader,
    unlabeled_dataloader=unlabeled_dataloader,
    checkpoint_path="{}/fcn_densenet169_student_hc_and_lc.pth".format(output_dir),
)
experiment_name = "Train 2nd student with high confidence and low confidence images - labeled ratio = {} %".format(
    100 - int(test_size * 100)
)

train_model(model, experiment_name, n_epochs)