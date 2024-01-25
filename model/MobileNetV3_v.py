import torch.nn as nn
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init



class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        expand_size =  max(in_size // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(expand_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_size, in_size, kernel_size=1, bias=False),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Bottleneck(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, act, se, stride):
        super(Bottleneck, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = act(inplace=True)

        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = act(inplace=True)
        self.se = SeModule(expand_size) if se else nn.Identity()

        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.act3 = act(inplace=True)

        self.skip = None
        if stride == 1 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=in_size, kernel_size=3, groups=in_size, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(in_size),
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=True),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size == out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, groups=in_size, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

    def forward(self, x):
        skip = x

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))
        
        if self.skip is not None:
            skip = self.skip(skip)
        return self.act3(out + skip)
    

class MobileNetV3(nn.Module):
    def __init__(self, num_classes=1000, input_size=32, act=nn.Hardswish):
        super(MobileNetV3, self).__init__()

        if input_size == 224:
            s = 2  
        else:
            s = 1

        # Layer 0
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=s, padding=1, bias=False),
            nn.BatchNorm2d(16),
            act(inplace=True)
        )

        # Layer 1
        self.layer1 = nn.Sequential(
            Bottleneck(3, 16, 16, 16, nn.ReLU, True, s),
            Bottleneck(3, 16, 72, 24, nn.ReLU, False, 2),
            Bottleneck(3, 24, 88, 24, nn.ReLU, False, 1)
        )

        # Layer 2
        self.layer2 = nn.Sequential(
            Bottleneck(5, 24, 96, 40, act, True, 2),
            Bottleneck(5, 40, 240, 40, act, True, 1),
            Bottleneck(5, 40, 120, 48, act, True, 1),
            Bottleneck(5, 48, 144, 48, act, True, 1)
        )

        # Layer 3
        self.layer3 = nn.Sequential(
            Bottleneck(5, 48, 288, 96, act, True, 2),
            Bottleneck(5, 96, 576, 96, act, True, 1),
            Bottleneck(5, 96, 576, 96, act, True, 1)
        )

        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = act(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(576, num_classes)

        self.init_params()

    def init_params(self):
        # initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.hs2(self.bn2(self.conv2(out)))
        out = self.gap(out).flatten(1)
        out = self.fc(out)

        return out