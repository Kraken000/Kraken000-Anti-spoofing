import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

import os
import numpy as np
import matplotlib.pyplot as plt

from Module.mobilenetv2 import *

# Module of using GPU
################################################################################
def get_default_device():
    """Pick GPU if avaliable, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')


device = get_default_device()


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Numben of batches"""
        return len(self.dl)


################################################################################


image_train_transforms = transforms.Compose([
    # transforms.Resize(size=64),
    transforms.RandomResizedCrop(size=112, scale=(0.8, 1.0)),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(size=64),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

image_valid_transforms = transforms.Compose([
    transforms.Resize(size=112),
    transforms.CenterCrop(size=64),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

traindir = './dataset/train/'
# Datasets from folders
dataset = datasets.ImageFolder(root='./dataset/train/', transform=image_train_transforms)


# Module of picking a percent dataset as validation set
################################################################################
def split_indices(n, val_pct):
    # Determine size of validation set
    n_val = int(val_pct * n)
    # Create random permutation of 0 to n-1
    idxs = np.random.permutation(n)
    # Pick first n_val indices for validation set
    return idxs[n_val:], idxs[:n_val]


################################################################################
BS = 64

train_indices, val_indices = split_indices(len(dataset), val_pct=0.2)

# Training sampler and data loader
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
# train_dl = torch.utils.data.DataLoader(dataset, batch_size=BS, shuffle=True, num_workers=2)
train_dl = torch.utils.data.DataLoader(dataset, batch_size=BS, sampler=train_sampler)
train_dl = DeviceDataLoader(train_dl, device)

# Validation sampler and data loader
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
valid_dl = torch.utils.data.DataLoader(dataset, batch_size=BS, sampler=valid_sampler)
valid_dl = DeviceDataLoader(valid_dl, device)


################################################################################


def loss_batch(model, loss_func, xb, yb, opt=None, metric=None):
    """Calculate the loss and metric of a batch of data"""
    # Generate predictions
    preds = model(xb)
    # Calculate loss
    loss = loss_func(preds, yb)

    if opt is not None:
        # Compute gradients
        loss.backward()
        # Update parameters
        opt.step()
        # Reset gradients
        opt.zero_grad()

    metric_result = None
    if metric is not None:
        # Compute the metric
        metric_result = metric(preds, yb)

    return loss.item(), len(xb), metric_result


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


def fit(epochs, lr, model, loss_fn, train_dl, valid_dl, metric=None, opt_fn=None):
    losses, metrics = [], []

    # Instantiate the optimizer
    if opt_fn is None: opt_fn = torch.optim.SGD
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # Training
        for xb, yb in train_dl:
            loss, _, _ = loss_batch(model, loss_fn, xb, yb, opt)

        # Evaluation
        result = evaluate(model, loss_fn, valid_dl, metric)
        val_loss, total, val_metric = result

        # Record the loss & metric
        losses.append(val_loss)
        metrics.append(val_metric)

        # Print progress
        if metric is None:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, val_loss))
        else:
            print('Epoch [{}/{}], Loss: {:.4f}, {}:{:.4f}'.format(epoch + 1, epochs, val_loss, metric.__name__,
                                                                  val_metric))
    return losses, metrics


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item() / len(preds)


net = MobileNetV2()
to_device(net, device)

losses1, metrics1 = fit(3, 1e-4, net, F.cross_entropy, train_dl, valid_dl, accuracy)

# if __name__ == '__main__':
#     net = Module1()
#     to_device(net, device)
#
#     cirterion = nn.CrossEntropyLoss()
#
#     # optimizer = optim.Adam(net.parameters(), lr=1e-4)
#     optimizer = optim.SGD(net.parameters(), lr=1e-4)
#
#     for xb, yb in train_dl:
#         loss, length, _ = loss_batch(net, cirterion, xb, yb)
#         print("loss:", loss)
#         print("length:", length)
#         break
