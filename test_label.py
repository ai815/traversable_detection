import cv2
import os
import sys
import numbers
import csv
import torch
import numpy as np


def label_normal(label):

    ratio_x = 224/1280
    ratio_y = 224/720
    label[::2] = [l_o*ratio_x for l_o in label[::2]]
    label[1::2] = [l_e*ratio_y for l_e in label[1::2]]

    return label



with open('//DESKTOP-3DNOAGH/traversable_region_detection/train.csv','r') as f:
    lines = csv.reader(f)
    coors = list(lines)

# image = []
coors_list =[]
# image_path = []
for c in coors:
    img = cv2.resize(cv2.imread(c[0]), (224,224), interpolation=cv2.INTER_CUBIC)

    res = list(map(lambda sub: int(''.join([ele for ele in sub if ele.isnumeric()])), c[1:]))
    res = label_normal(res)
    res = [int(re) for re in res]
    res = [tuple(res[i:i + 2]) for i in range(0, len(res), 2)]

    print(res)
    cv2.line(img, res[0], res[1], (255, 255, 255), 15 // 2)
    cv2.line(img, res[0], res[2], (200, 200, 200), 15 // 2)
    cv2.imshow('resize', img)
    cv2.waitKey(0)

