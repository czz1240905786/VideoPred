import random
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
import cv2


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, ground_truth):
        for t in self.transforms:
            image, ground_truth = t(image, ground_truth)
        return image, ground_truth


class LoadImage(object):
    def __call__(self, image_dirlist, ground_truth_dir):
        image = 0
        for dirname in image_dirlist:
            if type(image) == int:
                image = cv2.imread(dirname)
                # preframe = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                tempframe = cv2.imread(dirname)
                image = np.append(image, tempframe, axis=2)

                # tempframe = cv2.cvtColor(tempframe, cv2.COLOR_BGR2GRAY)
                # ret, threshold = cv2.threshold(cv2.absdiff(preframe, tempframe), 20, 255, cv2.THRESH_BINARY)
                # preframe = tempframe
                # image = np.append(image, threshold.reshape((tempframe.shape[0], tempframe.shape[1], 1)), axis=2)
        ground_truth = cv2.imread(ground_truth_dir)
        return image, ground_truth


class ToTensor(object):
    def __call__(self, image, ground_truth):
        image = F.to_tensor(image)
        ground_truth = F.to_tensor(ground_truth)
        return image, ground_truth


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, ground_truth):
        image = F.normalize(image.float(), mean=self.mean, std=self.std)
        # print(image.float())
        ground_truth = ground_truth.float()  # / 256
        # image = image.float() / 256
        return image, ground_truth


class Normalize_test(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, ground_truth):
        image = F.normalize(image.float(), mean=self.mean, std=self.std)
        return image, ground_truth
