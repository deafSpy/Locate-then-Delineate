import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class EndToEndModel(nn.Module):
    def __init__(self, config):
        super(EndToEndModel, self).__init__()

        self.n_channels = config["in_channels"]
        self.n_classes = config["out_channels"]
        self.bilinear = config["bilinear"]
        self.ndim_non_img = config['quad_num']
        # print("n_channels:", n_channels.shape)
        # print("n_classes:", n_classes.shape)
        # print("bilinear:", bilinear.shape)
        # print("ndim_non_img:", ndim_non_img.shape)

        self.inc = DoubleConv(self.n_channels, 64)
        # print(self.inc.shape)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # contextual
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.contextual_fc = nn.Linear(1024, self.ndim_non_img)
    
        self.daft_block = DAFTBlock(1024, None, bottleneck_dim=512, ndim_non_img=self.ndim_non_img)
        
        self.up1 = Up(1024, self.bilinear)
        self.up_conv1 = DoubleConv(1024, 512)
        self.up2 = Up(512, self.bilinear)
        self.up_conv2 = DoubleConv(512, 256)
        self.up3 = Up(256, self.bilinear)
        self.up_conv3 = DoubleConv(256, 128)
        self.up4 = Up(128, self.bilinear)
        self.up_conv4 = DoubleConv(128, 64)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, img):
        x1 = self.inc(img)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        pooled = self.global_pool(x5).view(x5.size(0), -1)
        text_embed = self.contextual_fc(pooled)
        # print(text_embed.shape)

        x5 = self.daft_block(x5, text_embed)

        decode1 = self.up1(x5)
        x = concatenate_layers(decode1, x4)
        x = self.up_conv1(x)

        decode2 = self.up2(x)
        x = concatenate_layers(decode2, x3)
        x = self.up_conv2(x)

        decode3 = self.up3(x)
        x = concatenate_layers(decode3, x2)
        x = self.up_conv3(x)

        decode4 = self.up4(x)
        x = concatenate_layers(decode4, x1)
        x = self.up_conv4(x)

        logits = self.outc(x)
        return logits
