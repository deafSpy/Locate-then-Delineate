import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from .layers import *

class UnetDAFT(torch.nn.Module):
    def __init__(self, config):
        super(UnetDAFT, self).__init__()

        self.n_channels = config["in_channels"]
        self.n_classes = config["out_channels"]
        self.bilinear = config["bilinear"]
        factor = 2 if self.bilinear else 1

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up1 = Up(1024, self.bilinear)
        self.up_conv1 = DoubleConv(1024, 512)

        self.up2 = Up(512, self.bilinear)
        self.up_conv2 = DoubleConv(512, 256)

        self.up3 = Up(256, self.bilinear)
        self.up_conv3 = DoubleConv(256, 128)

        self.up4 = Up(128, self.bilinear)
        self.up_conv4 = DoubleConv(128, 64)

        self.outc = OutConv(64, self.n_classes)
        self.sigmoid = nn.Sigmoid()

        #Experimenting with DAFT blocks at multiple levels
        '''
        self.daft_block1 = DAFTBlock(64, None, bottleneck_dim=32)
        self.daft_block2 = DAFTBlock(128, None, bottleneck_dim=64)
        self.daft_block3 = DAFTBlock(256, None, bottleneck_dim=128)
        self.daft_block4 = DAFTBlock(512, None, bottleneck_dim=256)
        '''
        '''
        self.daft_block5 = DAFTBlock(1024, None, bottleneck_dim=512)
        self.daft_block6 = DAFTBlock(512, None, bottleneck_dim=256)
        self.daft_block7 = DAFTBlock(256, None, bottleneck_dim=128)
        self.daft_block8 = DAFTBlock(128, None, bottleneck_dim=64)
        '''

        self.daft_block = DAFTBlock(1024, None, bottleneck_dim=512, ndim_non_img=config['quad_num'])

    def forward(self, img, text_embed):

        x1 = self.inc(img)
        '''
        x2 = self.down1(self.daft_block1(x1,ids))
        x3 = self.down2(self.daft_block2(x2,ids))
        x4 = self.down3(self.daft_block3(x3,ids))
        x5 = self.down4(self.daft_block4(x4,ids))
        '''
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        #decode1 = self.up1(self.daft_block5(x5,ids))
        decode1 = self.up1(self.daft_block(x5,tabular_data))
        x = concatenate_layers(decode1, x4)
        x = self.up_conv1(x)

        #decode2 = self.up2(self.daft_block6(x,ids))
        decode2 = self.up2(x)
        x = concatenate_layers(decode2, x3)
        x = self.up_conv2(x)

        #decode3 = self.up3(self.daft_block7(x,ids))
        decode3 = self.up3(x)
        x = concatenate_layers(decode3, x2)
        x = self.up_conv3(x)

        #decode4 = self.up4(self.daft_block8(x,ids))
        decode4 = self.up4(x)
        x = concatenate_layers(decode4, x1)
        x = self.up_conv4(x)

        logits = self.outc(x)

        return logits
