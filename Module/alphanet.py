import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Alphanet(nn.Module):
    def __init__(self):
        super(Alphanet, self).__init__()

        # first CONV => RELU => CONV => RELU => POOL layer set
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),

            # second CONV => RELU => CONV => RELU => POOL layer set
            nn.Conv2d(16, 32, 3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2)
        )
        # first (and only) set of FC => RELU layers
        self.fc = nn.Sequential(
            nn.Linear(32 * 12 * 12, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.conv(x)

        # print("x.shape:", x.shape)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


if "__name__" == "__main__":
    net = Alphanet()
    x = Variable(torch.randn(2,3,32,32))
    y = net(x)
    print(y.size())