import argparse
import json
import os
import numpy as np
import torch
from torchvision.transforms import Normalize as Normalize_th
# mean=torch.tensor([0.485]*6)
# std=0.229
# label = [234,34,56,78,9,90]
#
# label = (torch.from_numpy(np.array(label)).type(torch.float) / 255).type(torch.float)
# label = (label-mean)/std
# print(label)
#
# print((label*std + mean)*225, len(label))
#
#
# mean = torch.as_tensor(mean)
# std = torch.as_tensor(std)
# std_inv = 1 / (std + 1e-7)
# mean_inv = -mean * std_inv

import shutil
import time
import datetime
from region_detection import *

from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from config import *
from dataset import *
from lane_net import *

from utils.transforms.transforms import *
from utils.transforms.data_argumentation import *
from utils.lr_scheduler import PolyLR


#Rotation(2),
SAVE_FOLDER = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S").replace(':', '-')
if os.path.exists('runs/' + str(SAVE_FOLDER)) is False:
    os.makedirs('runs/' + str(SAVE_FOLDER))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, default='')
    parser.add_argument('--resume', '-r', action='store_true')
    args = parser.parse_args()
    return args


args = parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)
#
# transform_train = Compose(Resize(224), Darkness(30), Rotation(2), ToTensor())
transform_train = Compose(Resize(224),ToTensor())

train_dataset = Tusimple(Dataset_Path['my_image'], "train", transform_train)
train_loader = DataLoader(train_dataset, collate_fn=train_dataset.collate, batch_size=10, shuffle=True)

net = LaneNet(pretrained=True)
net = net.to(device)
# net = torch.nn.DataParallel(net)

# for name, para in net.named_parameters():
#     para.requires_grad = True
#     print(name, para)
# optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr = 0.001, weight_decay=0)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
# lr_scheduler = PolyLR(optimizer, 0.9, 10)

best_val_loss = 0.1

loss_mse = nn.MSELoss()
def loss_cal(output, label):
    # _____________________________________________________
    diff = np.array(np.array(label)-np.array(output))
    diff__ = []
    for d in diff:
        diff_ = []
        d = np.square(d)
        for i in range(0, len(d), 2):
            diff_.append(np.sqrt(np.sum(np.square(d[i:i+2]))))
        diff__.append(diff_)
    # print("diff__", diff__)
    sum_col = np.sum(diff__, axis=0)

    loss_0 = sum_col[0]
    loss_1 = sum_col[1]
    loss_2 = sum_col[2]
    loss = (loss_0 + loss_1 + loss_2) / len(output)
    return loss

def train(epoch):
    net.train()
    train_loss = 0
    for batch_idx, sample in enumerate(train_loader):
        img = sample['img'].to(device)
        # print(img)
        # label = torch.tensor(sample['segLabel']).float()
        label = torch.tensor(sample['segLabel']).clone().detach().requires_grad_(True)
        print(len(img), len(label))

        optimizer.zero_grad()
        output = net(img)
        output = output.squeeze(1)
        output = output.squeeze(1)

        output = Variable(output)
        label = Variable(label, requires_grad=False)

        print('label', len(label), label)
        print('output', len(output), output)

        # loss = Variable(torch.tensor(loss_cal(output, label)), requires_grad=True)
        loss = Variable(torch.tensor(loss_mse(output, label)), requires_grad=True)
        print(loss)

        # train_loss = train_loss+loss
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()



    if epoch % 5 == 0:
        save_dict = {
            "epoch": epoch,
            "net": net.module.state_dict() if isinstance(net, torch.nn.DataParallel) else net.state_dict(),
            "optim": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict()
        }

        torch.save(save_dict, os.path.join('runs', SAVE_FOLDER,
                                           'epoch{0}-loss.pth'.format(epoch, float(train_loss))))
    lr_scheduler.step()
    print("------------------------\n")

# ____________________________________________________________________________________________________________________________________________________________

transform_val = Compose(Resize(224), ToTensor())
val_dataset = Tusimple(Val_Path['val_image'], "val", transform_val)
val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=val_dataset.collate, num_workers=0)
exp_dir = 'runs/'


def val(epoch):
    global best_val_loss
    net.eval()
    val_loss = 0
    with torch.no_grad():
        for id, sample in enumerate(val_loader):
            img = sample['img'].to(device)
            label = sample['segLabel'].clone().detach().requires_grad_(True)
            output = net(img)
            output = output.squeeze(1)
            output = output.squeeze(1)

            output = Variable(output)
            label = Variable(label, requires_grad=False)

            # loss = Variable(torch.tensor(loss_cal(output, label)), requires_grad=True)
            loss = Variable(torch.tensor(loss_mse(output, label)), requires_grad=True)

            if isinstance(net, torch.nn.DataParallel):
                loss = loss.sum()
            print(loss)
            val_loss += loss.item()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_name = os.path.join(exp_dir, '_best.pth')
        copy_name = os.path.join(exp_dir,  'val_best.pth')
        shutil.copyfile(save_name, copy_name)
# ____________________________________________________________________________________________________________________________________________________________

def main():
    # run function train and val
    global best_val_loss
    if args.resume:
        # save_dict, os.path.join('runs', SAVE_FOLDER,
        #                         'epoch{0}-loss-fine.pth'.format(epoch, float(train_loss)))
        save_dict = torch.load(os.path.join(exp_dir, exp_dir.split('/')[-1] + '.pth'))
        if isinstance(net, torch.nn.DataParallel):
            net.module.load_state_dict(save_dict['net'])
        else:
            net.load_state_dict(save_dict['net'])
        optimizer.load_state_dict(save_dict['optim'])
        lr_scheduler.load_state_dict(save_dict['lr_scheduler'])
        start_epoch = save_dict['epoch'] + 1
        best_val_loss = save_dict.get("best_val_loss", 1)
    else:
        start_epoch = 0

    for epoch in range(start_epoch, 2000):
        train(epoch)
        if epoch % 5 == 0:
            print("\nValidation For Experiment: ", exp_dir)
            print(time.strftime('%H:%M:%S', time.localtime()))
            val(epoch)


if __name__ == '__main__':
    main()
