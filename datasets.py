import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from hyperparams import *


class PolypDataset(Dataset):
    """
    dataloader for polyp segmentation tasks
    """

    def __init__(self, image_paths=[], gt_paths=[], trainsize=352, transform=None):
        self.trainsize = trainsize
        self.images = image_paths
        self.masks = gt_paths
        self.size = len(self.images)
        self.transform = transform

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        mask = self.binary_loader(self.masks[index])

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            mask = mask / 255

        sample = dict(
            image=image,
            mask=mask.unsqueeze(0),
            image_path=self.images[index],
            mask_path=self.masks[index],
        )

        return sample

    def rgb_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f).resize((trainsize, trainsize), Image.LINEAR)
            return np.array(img.convert("RGB"))

    def binary_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f).resize((trainsize, trainsize), Image.NEAREST)
            img = np.array(img.convert("L"))
            _, im_th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            return im_th

    def __len__(self):
        return self.size
