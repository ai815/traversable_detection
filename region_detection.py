import cv2
import os
import sys
import numbers
import csv
import torch
import numpy as np


def label_normal(label):
    # mean = torch.tensor([1000.] * 6)
    # std = 500.
    # # normalize the label to range(-1, 1)
    # label = (torch.from_numpy(np.array(label)).type(torch.float)).type(torch.float)
    # label = (label - mean) / std
    # print(label, label[::2], label[1::2])
    # label[::2] = [l_o/1280 for l_o in label[::2]]
    # label[1::2] = [l_e/720 for l_e in label[1::2]]
    # ratio_x = 224/1280
    # ratio_y = 224/720
    # label[::2] = [(l_o*ratio_x - 112) / 112 for l_o in label[::2]]
    # label[1::2] = [(l_e*ratio_y - 112)/112 for l_e in label[1::2]]
    label[::2] = [(l_o-640)/640 for l_o in label[::2]]
    label[1::2] = [(l_e-350)/350 for l_e in label[1::2]]
    # print(label)
    label = (torch.from_numpy(np.array(label)).type(torch.float)).type(torch.float)
    return label


with open('//DESKTOP-3DNOAGH/traversable_region_detection/test.csv','r') as f:
    lines = csv.reader(f)
    coors = list(lines)

image = []
coors_list =[]
image_path = []
for c in coors:
    # print(c)
    image.append(cv2.imread(c[0]))
    # img = cv2.resize(cv2.imread(c[0]), (1280,720), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('resize', img)
    # cv2.waitKey(0)
    res = list(map(lambda sub: int(''.join([ele for ele in sub if ele.isnumeric()])), c[1:]))
    # segLabel = torch.from_numpy(np.array([res[i:i+2] for i in range(0, len(res),2)])).type(torch.long)
    # coors_list.append([res[i:i+2] for i in range(0, len(res),2)])
    res = label_normal(res)
    coors_list.append(res)
    image_path.append(c[0])

img_label = list(zip(image, coors_list, image_path))


