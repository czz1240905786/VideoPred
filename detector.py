import torch
import torch.nn as nn
import torch.nn.functional as F


class Detector(nn.Module):
    def __init__(self, join_num):
        """
        *****[[[  remember to write version.txt after changing net structure  ]]]****
        :param join_num:num of input picture
        """
        super(Detector, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3*join_num, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1)
        self.activation = lambda x: torch.sigmoid(x)

    def forward(self, x):
        # print(x.shape)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        return x
