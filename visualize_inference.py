import glob
import os

import torch
import wandb
from datasets import PolypDataset
from hyperparams import (
    trainsize, project, wandb_host, wandb_key
)

from transforms import val_transform

from utils import soft_predict, UnNormalize


os.environ["WANDB_API_KEY"] = wandb_key
os.environ["WANDB_BASE_URL"] = wandb_host

# Login wandb
wandb.login()
unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def get_testdataset(dataset):
    X = glob.glob("data/TestDataset/{}/images/*.png".format(dataset))
    y = glob.glob("data/TestDataset/{}/masks/*.png".format(dataset))

    return PolypDataset(X, y, trainsize=trainsize, transform=val_transform)


models = []
model_names = [
    'teacher', 'student', 'student_hc', 'student_hc_and_lc'
]

for name in model_names:
    model = torch.load("pretrained/fcn_densenet169_{}.pth".format(name))
    models.append(model)

datasets = os.listdir("data/TestDataset/")

for dataset in datasets:
    test_dataset = get_testdataset(dataset)
    # Predict with TTA in test set
    run = wandb.init(project=project, name="Testset_compare_result")

    # Save image to table
    artifact = wandb.Artifact(name="Test_{}".format(dataset), type="raw_data")

    columns = ["Raw", "Ground Truth"]

    columns.extend(model_names)

    table = wandb.Table(
        columns=columns
    )

    res = []

    for sample in test_dataset:
        image = sample['image']
        gt_mask = sample['mask']

        table_data = [
            wandb.Image(unorm(image)),
            wandb.Image(gt_mask),
        ]

        for model in models:
            prob_mask = soft_predict(model, torch.unsqueeze(image, 0).float())
            pred_mask = (prob_mask > 0.5).float()
            table_data.append(wandb.Image(pred_mask))

        table.add_data(*table_data)

    artifact.add(table, "test_sample")

    run.log_artifact(artifact)
    run.finish()