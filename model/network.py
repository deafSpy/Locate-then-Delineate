import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from benchmarks.contextual_net import *
from .unet_daft import *

class MyNetwork(torch.nn.Module):
    def __init__(self, config):
        super(MyNetwork, self).__init__()

        state_dict = torch.load(config["pretrained_unet"])["state_dict"]
        state_dict_copy = OrderedDict()

        for key in state_dict.keys():
            new_key = key[6:]
            state_dict_copy[new_key] = state_dict[key].clone()

        self.contextualnet = CONTEXTUALNET(config)
        self.contextualnet.load_state_dict(state_dict_copy, strict=False)
        self.maxpool = nn.MaxPool2d((config["img_size"]//2,config["img_size"]//2)) #4 quadrants
        '''
        self.maxpool = nn.MaxPool2d((config["img_size"]//3,config["img_size"]//2)) #6 regions
        '''
        self.unet_daft = UnetDAFT(config)
        self.unet_daft.load_state_dict(state_dict_copy, strict=False)

    def forward(self, img, text_embed):
        out = self.contextualnet(img, text_embed)
        tabular_data = self.maxpool(out)
        out = self.unet_daft(img, tabular_data)
        return out
