import numpy as np
import segmentation_models_pytorch as smp
import torch
import pickle
from tqdm import tqdm
from collections import defaultdict
from hyperparams import device


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(
            torch.stack(self.losses[np.maximum(len(self.losses) - self.num, 0):])
        )


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def predict(model, image):
    image = image.to(device)
    logits = model(image)
    prob_mask = logits.sigmoid()
    pred_mask = (prob_mask > 0.5).float()
    return pred_mask


def soft_predict(model, image):
    with torch.no_grad():
        logit = model(image.cuda())
        return logit.sigmoid()

def calculate_iou(best_mask, pred_mask):
    tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), best_mask.long(), mode="binary")
    per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
    return per_image_iou


# Util function for prioritizing stable images
def generate_image_scores(best_model, checkpoint_paths, unlabeled_dataset):
    image_scores = defaultdict(list)

    # Calculate all ious for each image
    for path in tqdm(checkpoint_paths):
        model = torch.load(path)
        model.to(device)

        for i in range(len(unlabeled_dataset)):
            image = unlabeled_dataset.__getitem__(i)['image']
            image_path = unlabeled_dataset.__getitem__(i)['image_path']
            best_mask = predict(best_model, image.unsqueeze(0))
            current_mask = predict(model, image.unsqueeze(0))
            iou_score = calculate_iou(current_mask, best_mask)

            image_scores[image_path].append(iou_score)

    # Calculate mean score
    image_mean_score = {}

    for key, value in image_scores.items():
        mean = torch.mean(torch.stack(value)).cpu().numpy()
        image_mean_score[key] = float(mean)
    return image_mean_score


def pickle_dump(obj, filename='image_mean_score.pkl'):
    with open(filename, 'wb') as fp:
        pickle.dump(obj, fp)


def pickle_load(filename='image_mean_score.pkl'):
    with open(filename, 'rb') as fp:
        obj = pickle.load(fp)
    return obj


def prioritizer(score_dict, confidence_threshold):
    high_confidence_images = [k for k, v in score_dict.items() if v >= confidence_threshold]
    low_confidence_images = [k for k, v in score_dict.items() if v < confidence_threshold]
    return high_confidence_images, low_confidence_images