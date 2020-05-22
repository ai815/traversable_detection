import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from region_detection import *

class Tusimple(Dataset):
    """
    image_set is splitted into three partitions: train, val, test.
    train includes label_data_0313.json, label_data_0601.json
    val includes label_data_0531.json
    test includes test_label.json
    """

    def __init__(self, path, image_set, transforms=None):
        print(path)
        super(Tusimple, self).__init__()
        # assert image_set in ('train', 'val', 'test'), "image_set is not valid!"
        self.data_dir_path = path
        self.image_set = image_set
        self.transforms = transforms

        self.createIndex()

    def createIndex(self):
        self.img_list = image_path
        self.segLabel_list = coors_list

    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        sample = {'img': img,
                  'segLabel': self.segLabel_list[idx],
                  'img_name': self.img_list[idx]}

        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __len__(self):
        return len(self.img_list)


    @staticmethod
    def collate(batch):
        if isinstance(batch[0]['img'], torch.Tensor):
            img = torch.stack([b['img'] for b in batch])
        else:
            img = [b['img'] for b in batch]

        if batch[0]['segLabel'] is None:
            segLabel = None
        elif isinstance(batch[0]['segLabel'], torch.Tensor):
            segLabel = torch.stack([b['segLabel'] for b in batch])
        else:
            segLabel = [b['segLabel'] for b in batch]

        samples = {'img': img,
                   'segLabel': segLabel,
                   'img_name': [x['img_name'] for x in batch]}

        return samples