#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 22:20:16 2018

@author: Taha Emara  @email: taha@emaraic.com
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
import math


from liteseg_model.backbone_networks import darknet
from liteseg_model import aspp
from liteseg_model.separableconv import SeparableConv2d 

class RT(nn.Module):
    
    def __init__(self,  n_classes=19, pretrained=True,PRETRAINED_WEIGHTS="."):
       
        super(RT, self).__init__()
        print("LiteSeg-DarkNet...")
        if pretrained:
            self.resnet_features=darknet.Darknet19(weights_file=PRETRAINED_WEIGHTS)
        else:
            self.resnet_features=darknet.Darknet19(weights_file=None)
        rates = [1, 3, 6, 9]

        self.aspp1 = aspp.ASPP(1024, 96, rate=rates[0])
        self.aspp2 = aspp.ASPP(1024, 96, rate=rates[1])
        self.aspp3 = aspp.ASPP(1024, 96, rate=rates[2])
        self.aspp4 = aspp.ASPP(1024, 96, rate=rates[3])

        self.relu = nn.ReLU()
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(1024, 96, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(96),
                                             nn.ReLU())
        
        #self.conv1 = nn.Conv2d(1504, 96, 1, bias=False)#480 1504
        self.conv1 =SeparableConv2d(1504,96,1)
        self.bn1 = nn.BatchNorm2d(96)

        # adopt [1x1, 48] for channel reduction.
        #self.conv2 = nn.Conv2d(128, 32, 1, bias=False)#128 for no previous feature 7  ,64 ---3
        self.conv2 = SeparableConv2d(128,32,1)
        self.bn2 = nn.BatchNorm2d(32)
#    
        self.last_conv = nn.Sequential(#nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=1, bias=False),
                                       SeparableConv2d(128,96,3,1,1),
                                       nn.BatchNorm2d(96),
                                       nn.ReLU(),
                                       #nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=False),
                                       SeparableConv2d(96,96,3,1,1),
                                       nn.BatchNorm2d(96),
                                       nn.ReLU(),
                                       nn.Conv2d(96, n_classes, kernel_size=1, stride=1))
 
    def forward(self, input):
        x, low_level_features = self.resnet_features(input)
        #print('x ',x.size(),' low features',low_level_features.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        #print('x1',x1.size())
        x = torch.cat((x,x1, x2, x3, x4, x5), dim=1)#x = torch.cat((x,x1, x2, x3, x4, x5), dim=1)

        #ablation=torch.max(x, 1)[1]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(int(math.ceil(input.size()[-2]/4)),##/4 --7  /2 ---3
                                int(math.ceil(input.size()[-1]/4))), mode='bilinear', align_corners=True)
        #ablation=torch.max(x, 1)[1]
        #print('x',x.size())

        ##comment to remove low feature
        #print('low level features',low_level_features.size())
        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)
        x = torch.cat((x, low_level_features), dim=1)

        
        #ablation=torch.max(x, 1)[1]
        x = self.last_conv(x)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x #,ablation

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
