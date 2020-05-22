import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class LaneNet(nn.Module):
    def __init__(
            self,
            pretrained = True,
            **kwargs
    ):
        super(LaneNet, self).__init__()
        self.pretrained = pretrained
        self.net_init()

    def net_init(self):

        # -------------------------------------------------------------------------------------

        # self.backbone = models.vgg19_bn(pretrained=self.pretrained).features
        # # print(self.backbone)
        # self.points = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 512),
        #     nn.PReLU(),
        #     nn.Dropout(),
        #     nn.Linear(512,512),
        #     nn.Dropout(),
        #     nn.Linear(512, 6)
        # )
        # self.backbone = models.vgg16_bn(pretrained=self.pretrained).features
        # # ----------------- process backbone -----------------
        # for i in [34, 37, 40]:
        #     conv = self.backbone._modules[str(i)]
        #     dilated_conv = nn.Conv2d(
        #         conv.in_channels, conv.out_channels, conv.kernel_size, stride=conv.stride,
        #         padding=tuple(p * 2 for p in conv.padding), dilation=2, bias=(conv.bias is not None)
        #     )
        #     dilated_conv.load_state_dict(conv.state_dict())
        #     self.backbone._modules[str(i)] = dilated_conv
        # self.backbone._modules.pop('33')
        # self.backbone._modules.pop('43')
        #
        # # ----------------- additional conv -----------------
        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(in_channels=512, out_channels=1024, kernel_size= 3, stride=2,dilation=4, bias=True),
        #     nn.BatchNorm2d(1024),
        #     nn.PReLU(),
        #     nn.Conv2d(1024, 128, 3, stride=2, bias=True),
        #     nn.BatchNorm2d(128),
        #     nn.PReLU(),
        #     nn.Conv2d(128, 32, 3, stride=2, padding=2, bias=True),
        #     nn.BatchNorm2d(32),
        #     nn.PReLU(),
        #     nn.Conv2d(32, 1, 3, stride=2,bias=True),
        #     nn.BatchNorm2d(1),
        #     nn.PReLU(),
        #     nn.Linear(1,6),
        # )
        self.backbone = models.vgg16_bn(pretrained=self.pretrained).features
        # ----------------- process backbone -----------------
        # for i in [34, 37, 40]:
        #     conv = self.backbone._modules[str(i)]
        #     dilated_conv = nn.Conv2d(
        #         conv.in_channels, conv.out_channels, conv.kernel_size, stride=conv.stride,
        #         padding=tuple(p * 2 for p in conv.padding), dilation=2, bias=(conv.bias is not None)
        #     )
        #     dilated_conv.load_state_dict(conv.state_dict())
        #     self.backbone._modules[str(i)] = dilated_conv
        # self.backbone._modules.pop('33')
        # self.backbone._modules.pop('43')
        #
        # ----------------- additional conv -----------------
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size= 3, padding = 2, stride=2,dilation=1, bias=True),
            nn.BatchNorm2d(1024),
            nn.PReLU(),
            nn.Conv2d(1024, 128, 3, stride=1, bias=True),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 32, 3, stride=2, bias=True),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            # nn.Conv2d(32, 1, 3, stride=2,bias=True),
            # nn.BatchNorm2d(1),
            # nn.PReLU(),
            nn.Linear(1,6),
        )
    def forward(self, x):
        fc1 = nn.Linear(32*1*6, 6)
        x = self.backbone(x)
        output = self.layer1(x)
        # print(output.shape)
        output = output.view(-1, 32*1*6)
        print(x.shape)
        output = fc1(output)
        print(output.shape)
        # ---------------------------------------
        # x = self.backbone(x)
        # x = x.view(-1, 512 * 7 * 7)
        # x = self.points(x)
        # output = x
        # print(output)
        # ------------------------------------------
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x)))
        # x = x.view(-1, 64 * 12 * 12)
        # x = F.relu(self.fc1(self.dropout(x)))
        # output = self.fc2(self.dropout(x))
        return output

