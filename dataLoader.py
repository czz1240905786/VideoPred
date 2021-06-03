import os

import cv2
import torch
from torch.utils import data
from torchvision.transforms import functional as F

import numpy as np

import transforms


class MyDataSet(data.Dataset):
    def __init__(self, root, mode, join_num):
        assert mode in ['train', 'test']

        self.root = root
        self.join_num = join_num
        self.listdir = os.listdir(root)
        self.item = []

        for dirname in self.listdir:
            self.item.append(len(os.listdir(root+"\\"+dirname)))

        self.transforms = transforms.Compose([
            transforms.LoadImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.0, std=1.0)
        ]) if mode == 'train' else \
            transforms.Compose([
                transforms.LoadImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.0, std=1.0)
            ])

    def __len__(self):
        length = 0
        for num in self.item:
            length += num-self.join_num
        return length

    def __getitem__(self, item):
        temp = 0
        # print(self.item)
        while item >= self.item[temp]-self.join_num:
            item -= self.item[temp]-self.join_num
            temp += 1

        dirname = self.root + "\\" + self.listdir[temp] + "\\"
        imgdir_list = list()
        for i in range(self.join_num):
            imgdir_list.append(dirname+str(i+item)+".jpg")
        label_dir = dirname+str(self.join_num+item)+".jpg"

        return self.transforms(imgdir_list, label_dir), label_dir


class DDataSet(data.Dataset):
    def __init__(self, root, join_num):
        self.root = root
        self.listdir = os.listdir(root)
        self.item = []
        self.jn = join_num

        for dirname in self.listdir:
            self.item.append(os.listdir(root + "\\" + dirname)[join_num:])

        self.transforms = transforms.Compose([
            transforms.LoadImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.0, std=1.0)
        ])

    def __len__(self):
        length = 0
        for line in self.item:
            length += len(line)
        return length

    def __getitem__(self, item):
        temp = 0
        while item >= len(self.item[temp]):
            item -= len(self.item[temp])
            temp += 1

        img_dir = self.root + "\\" + self.listdir[temp] + "\\" + self.item[temp][item]
        pre_list = list()
        for i in range(self.jn):
            pre_list.append(f"{self.root}\\{self.listdir[temp]}\\{item//2+i}.jpg")
        return self.transforms(pre_list, img_dir), img_dir


# if __name__ == "__main__":
#     batch = 1
#     trainloader = data.DataLoader(MyDataSet(root='data', mode='train', join_num=3), batch_size=batch, shuffle=True, num_workers=2)
#     testloader = data.DataLoader(MyDataSet(root='data', mode='test', join_num=3), batch_size=batch, num_workers=0)
# #     a = 0
# #
#     for (img, gt), name in trainloader:
#         # a += 1
#         print(img.shape)
#         print(gt.shape)
#         print(torch.view())
#         print(name)
#         origin = (gt[0].numpy()*255).transpose((1, 2, 0)).astype(np.uint8)
#
#         print(origin.shape)
#         # origin = cv2.cvtColor(origin, cv2.COLOR_RGB2BGR)
#         # cv2.namedWindow("show")
#         cv2.imshow("show", origin)
#         gtt = cv2.imread(name[0])
#         cv2.imshow("gtt", gtt)
#         print(gtt)
#         print(origin)
#         at = (gtt - origin)
#         cv2.waitKey()
#
#         pass
#
#     print(a)
