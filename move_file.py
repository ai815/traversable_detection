import os
import csv
import shutil

val = 'val.csv'
test = 'test.csv'
move_to_val = "//DESKTOP-ASB277M/Data_3D/Training/3D_data/val/"
move_to_test = "//DESKTOP-ASB277M/Data_3D/Training/3D_data/test/"

def move_file(move_to, file):
    with open(file,'r') as f:
        lines = csv.reader(f)
        coors = list(lines)

    for c in coors:
        print(c[0])
        shutil.move(c[0], os.path.join(move_to, os.path.basename(c[0])))

move_file(move_to_val, val)
move_file(move_to_test, test)