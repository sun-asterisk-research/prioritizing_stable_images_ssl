import glob
import os

import torch
from torch.utils.data import DataLoader

from datasets import PolypDataset
from hyperparams import (
    trainsize,
)
from tabulate import tabulate
from utils import AvgMeter
import pandas as pd

# Import model
from models import PrioritizingStableSegmentationModel
import pytorch_lightning as pl
from transforms import val_transform

# Import model


def get_testdataset(dataset):
    X = glob.glob("data/TestDataset/{}/images/*.png".format(dataset))
    y = glob.glob("data/TestDataset/{}/masks/*.png".format(dataset))

    return PolypDataset(X, y, trainsize=trainsize, transform=val_transform)


n_cpu = os.cpu_count()
model = PrioritizingStableSegmentationModel(
    "FPN", "densenet169", in_channels=3, out_classes=1, checkpoint_path=""
)
model.model = torch.load("pretrained/fcn_densenet169_student_hc_and_lc.pth")

trainer = pl.Trainer(gpus=1, max_epochs=100)

datasets = os.listdir("data/TestDataset/")
table = []
headers = ["Dataset", "IoU", "Dice"]
ious, dices = AvgMeter(), AvgMeter()

# datasets = ['Kvasir']

for dataset in datasets:
    tmp_dataset = get_testdataset(dataset)
    tmp_dataloader = DataLoader(
        tmp_dataset, batch_size=32, shuffle=False, num_workers=n_cpu
    )
    test_metrics = trainer.test(model, dataloaders=tmp_dataloader, verbose=False)

    iou = test_metrics[0]["test_per_image_iou"]
    dice = test_metrics[0]["test_per_image_dice"]
    ious.update(iou)
    dices.update(dice)

    table.append([dataset, iou, dice])

table.append(["Total", ious.avg, dices.avg])
df = pd.DataFrame(table)
df.to_csv("result.csv")
print(tabulate(table, headers=headers, tablefmt="fancy_grid"))
