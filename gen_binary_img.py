import cv2
import csv
import os
import numpy as np

with open('E:/traversable_region_detection/train.csv','r') as f:
    lines = csv.reader(f)
    coors = list(lines)


image_path = []
i = 0
for ele in coors:
    # print(c)
    # img = cv2.imread(ele[0])
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    res = list(map(lambda sub: int(''.join([ele for ele in sub if ele.isnumeric()])), ele[1:]))
    points = [tuple(res[i:i+2]) for i in range(0, len(res),2)]
    # print(points)
    seg_img = np.zeros((720, 1280, 3))
    cv2.line(seg_img, points[0], points[1], (255, 255, 255), 15 // 2)
    cv2.line(seg_img, points[0], points[2], (200, 200, 200), 15 // 2)
    # i=i+1

    # for r in range(0, gray.shape[0]):
    #     for c in range(0, gray.shape[1]):
    #         if [c, r] in points:
    #             gray[r-10:r+10,c-5:c+5] = 255
    #             # print(r,c)
    #         else:
    #             gray[r,c] = 0
    #
    #
    image_path = ele[0]
    # print(image_path)
    cv2.imwrite('E:/traversable_region_detection/seg_img/'+os.path.basename(image_path), seg_img)
    # cv2.imshow('binary', seg_img)
    # cv2.waitKey(0)