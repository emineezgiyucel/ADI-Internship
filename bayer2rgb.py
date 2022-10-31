import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from torch import nn
import torchvision
from torchvision import transforms
import ai8x
import torch.optim as optim
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as trans
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data.dataloader import DataLoader as LoadData
import torch.optim as optim
from torch.utils.data import Dataset
import cv2
from skimage import io
import torchvision

class bayer2rgbnet(nn.Module):
    def __init__(self,num_classes=None,  # pylint: disable=unused-argument
            num_channels=1,
            dimensions=(128, 128),  # pylint: disable=unused-argument
            bias=False,
            **kwargs):
        super().__init__()
        self.l1 = ai8x.Conv2d(4, 3, kernel_size=1, padding=0, bias=False)
        self.l2 = ai8x.ConvTranspose2d(3, 3, kernel_size=3, padding=1, stride=2, bias=False)
        # comment out for folded+trans+conv
        #self.l3 = ai8x.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
        # comment out for folded+b2rgb
        #self.conv1 = ai8x.FusedConv2dReLU(3, 16, 3, padding=1, bias=False)
        #self.conv1_2 = ai8x.FusedConv2dReLU(16, 32, 3, padding=1, bias=False)
        #self.conv1_3 = ai8x.FusedConv2dReLU(32, 64, 3, padding=1, bias=False)
        #self.conv1_4 = ai8x.FusedConv2dReLU(64, 128, 3, padding=1, bias=False)
        #self.conv2 = ai8x.FusedConv2dReLU(128, 64, 1, padding=0, bias=False)
        #self.conv3 = ai8x.FusedConv2dReLU(64, 32, 3, padding=1, bias=False)
        #self.conv3_2 = ai8x.Conv2d(32, 3, 3, padding=1, bias=False)
        self._init_layers()

    def _init_layers(self):
        #Initialize Weights
        self.l1.op.weight.data[0, :, 0, 0] = torch.tensor([0, 0, 1, 0])
        self.l1.op.weight.data[1, :, 0, 0] = torch.tensor([0.5, 0, 0, 0.5])
        self.l1.op.weight.data[2, :, 0, 0] = torch.tensor([0, 1, 0, 0])
        self.l1.op.weight.requires_grad = False

    def forward(self,x):
        x = self.l1(x)
        x = self.l2(x)
        # comment out for folded+trans+conv
        #x = self.l3(x)
        # comment out for folded+b2rgb
        #x = self.conv1(x)
        #x = self.conv1_2(x)
        #x = self.conv1_3(x)
        #x = self.conv1_4(x)
        #x = self.conv2(x)
        #x = self.conv3(x)
        #x = self.conv3_2(x)
        return x
    
def b2rgb(pretrained=False, **kwargs):
    assert not pretrained
    return bayer2rgbnet(**kwargs)


models = [
    {
        'name': 'bayer2rgbnet',
        'min_input': 1,
        'dim': 2,
    },
]


