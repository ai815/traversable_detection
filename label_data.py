import cv2
import os
import argparse
import glob
from os.path import isfile, join

image_path = '//DESKTOP-ASB277M/Data_3D/Training/3D_data/images/'
image_label = '//DESKTOP-ASB277M/Data_3D/Training/3D_data/labels_region/'
def get_coords_by_mouse(event, x,y, flags, param):
    if event ==cv2.EVENT_LBUTTONDOWN:
        xy = '%d,%d'%(x,y)
        a.append([x,y])
        cv2.circle(img, (x,y), 1,(255, 0 ,0), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0,(0,0,0),thickness=1)
        cv2.imshow('image', img)
        return a

def arg_parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--imgs', dest='image', default=image_path, help= 'imagesfile', type=str)
    ap.add_argument('--label', dest='label', default=image_label, help= 'imagelabel', type=str)
    return ap.parse_args()
args = arg_parse()

file = args.image

# files = [f for f in os.listdir(image_path) if isfile(join(image_path, f))]
# files.sort(key=lambda x : x[:-4][::-1])
files = []
for img in os.listdir(file):
    image_full_path = os.path.join(image_path, img)
    if img[-2:] == 'db':
        os.remove(image_full_path)
    if img[-4:] == '.png':
        files.append(image_full_path)
    # image = cv2.imread(image_full_path)

print(files)
a = []
image_name = []
for i in range(len(files)):
    filename = files[i]
    image_name.append(filename)
    img = cv2.imread(filename)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', get_coords_by_mouse)
    cv2.imshow('image', img)
    cv2.waitKey(0)
cv2.destroyAllWindows()

a_final = [a[i:i+3] for i in range(0, len(a), 3)]
name_coords_dic = dict(zip(image_name,a_final))
print(name_coords_dic)

with open('labels.csv', 'w', newline='') as f:
    for keys in name_coords_dic:
        f.write('%s,%s\n'%(keys,name_coords_dic[keys]))
