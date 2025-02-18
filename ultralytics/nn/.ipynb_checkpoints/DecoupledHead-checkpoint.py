import contextlib
from copy import deepcopy
from pathlib import Path
from ultralytics.nn.modules import (AIFI, C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x,Classify, Concat, Conv, Conv2, ConvTranspose, Detect, DWConv,DWConvTranspose2d,Focus, GhostBottleneck, GhostConv, HGBlock, HGStem, Pose, RepC3, RepConv,RTDETRDecoder, Segment,LightConv, RepConv, SpatialAttention)

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, caffe2_xavier_init, constant_init
#详细改进流程和操作，请关注B站博主：AI学术叫叫兽 

from mmcv.cnn import ConvModule

class DecoupledHead(nn.Module):
    def __init__(self, ch=256, nc=80,  anchors=()):
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.merge = Conv(ch, 256 , 1, 1)
        self.cls_convs1 = Conv(256 , 256 , 3, 1, 1)
        self.cls_convs2 = Conv(256 , 256 , 3, 1, 1)
        self.reg_convs1 = Conv(256 , 256 , 3, 1, 1)
        self.reg_convs2 = Conv(256 , 256 , 3, 1, 1)
        self.cls_preds = nn.Conv2d(256 , self.nc * self.na, 1)
        self.reg_preds = nn.Conv2d(256 , 4 * self.na, 1)
        self.obj_preds = nn.Conv2d(256 , 1 * self.na, 1)

    def forward(self, x):
        x = self.merge(x)
        x1 = self.cls_convs1(x)
        x1 = self.cls_convs2(x1)
        x1 = self.cls_preds(x1)
        x2 = self.reg_convs1(x)
        x2 = self.reg_convs2(x2)
        x21 = self.reg_preds(x2)
        x22 = self.obj_preds(x2)
        out = torch.cat([x21, x22, x1], 1)
        return out