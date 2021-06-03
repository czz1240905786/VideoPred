import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    """
    TODO:
    """
    def __init__(self, join_num):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(  # 3jn+3*640*480
            nn.Unflatten(1, (3, join_num+1)),  # 3*jn*640*480
            nn.Conv3d(in_channels=3, out_channels=16, kernel_size=(join_num+1, 5, 5), padding=(0, 2, 2)),  # 16*1*640*480
            # nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),  # 16*640*480
            nn.LeakyReLU(0.2),
            nn.Flatten(start_dim=1, end_dim=2),  # 16*640*480
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16*320*240
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),  # 32*320*240
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=4, stride=4),  # 32*80*60
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1),  # 1*80*60
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=4, stride=4),  # 1*20*15
            nn.Flatten(),
            nn.Linear(300, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x, y):
        """
        :param x: (batch * 3 * w * h)
        :param y: (batch * 3jn * w * h)
        """
        x = torch.cat((y, x), 1)
        x = self.dis(x)
        return x


class Generator(nn.Module):
    def __init__(self, join_num):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Unflatten(1, (3, join_num)),
            nn.Conv3d(in_channels=3, out_channels=3, kernel_size=(join_num, 11, 11), padding=(0, 5, 5)),
            nn.Flatten(start_dim=1, end_dim=2),
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),
            # nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, padding=2),
            # nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        :param x:batch, in_channel * join_num, W, H
        :return:
        """
        x = self.gen(x)
        return x
