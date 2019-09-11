import torch
import torch.nn as nn
import torch.utils.data

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import numpy as np

import logging
import os
import cv2

from Module.mobilenetv2 import *

logger = logging.getLogger(__name__)


# testdir = '/home/yjiang/Anti-spoofing/Anti_Spoofing_v1.0.0/dataset/test_demo/'
testdir = '/home/yjiang/Anti-spoofing/Anti_Spoofing_v1.0.0/dataset/test_demo_face'
model = MobileNetV2()

model.load_state_dict(torch.load('./checkpoint/mobilenetv2_v0.0.2_BS_64_IS_64_Adam.pkl',
                                 map_location=torch.device('cpu')))
model.eval()

BS = 4

test_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(testdir, transforms.Compose([
        transforms.Resize(112),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])),
    batch_size=BS,
    num_workers=4,
    pin_memory=True
)



count = 0
def Evaluate(test_loader, model):
    count = 0
    sum = 0

    for i, (input, target) in enumerate(test_loader):
        count += 1
        outputs = model(input)

        _, preds = torch.max(outputs, dim=1)

        sum += torch.sum(preds == target).item()

    print(sum)
    acc = sum / (count * BS)

        # sum += torch.sum(preds == target).item()
    print(count)

    return acc


res = Evaluate(test_loader, model)
print(res)





def evaluate(model, loss_fn, valid_dl, metric=None):
    """Calculate the total loss of validation set and required metric value"""
    with torch.no_grad():
        # Pass each batch through the model
        results = [loss_batch(model, loss_fn, xb, yb, metric=metric) for xb, yb in valid_dl]
        # Separate losses, counts and metrics
        losses, nums, metrics = zip(*results)
        # Total size of the dataset
        total = np.sum(nums)
        # Avg. loss across batches
        avg_loss = np.sum(np.multiply(losses, nums)) / total
        avg_metric = None
        if metric is not None:
            # Avg. of metric across batches
            avg_metric = np.sum(np.multiply(metrics, nums)) / total
    return avg_loss, total, avg_metric

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item() / len(preds)


