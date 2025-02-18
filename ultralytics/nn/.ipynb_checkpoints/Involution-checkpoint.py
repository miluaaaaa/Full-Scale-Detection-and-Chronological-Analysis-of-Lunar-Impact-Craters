import contextlib
from copy import deepcopy
from pathlib import Path
from ultralytics.nn.modules import (AIFI, C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x,Classify, Concat, Conv, Conv2, ConvTranspose, Detect, DWConv,DWConvTranspose2d,Focus, GhostBottleneck, GhostConv, HGBlock, HGStem, Pose, RepC3, RepConv,RTDETRDecoder, Segment,LightConv, RepConv, SpatialAttention)

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, caffe2_xavier_init, constant_init
#详细改进流程和操作，请关注B站博主：AI学术叫叫兽 

from mmcv.cnn import ConvModule

class Involution(nn.Module):

    def __init__(self,c1,c2,kernel_size,stride):
        super(Involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.c1 = c1
        reduction_ratio = 4
        self.group_channels = 16
        self.groups = self.c1 // self.group_channels
        self.conv1 = Conv(c1, c1 // reduction_ratio,1)
        self.conv2 = Conv(c1 // reduction_ratio,kernel_size**2 * self.groups,1,1)
           
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size-1)//2, stride)    

    def forward(self, x):

        weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size**2, h, w).unsqueeze(2)
       #out = _involution_cuda(x, weight, stride=self.stride, padding=(self.kernel_size-1)//2)
        #print("weight shape:",weight.shape)
        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size**2, h, w)
        #print("new out:",(weight*out).shape)
        out = (weight * out).sum(dim=3).view(b, self.c1, h, w)
     
        return out