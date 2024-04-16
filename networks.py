import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import *

class CLS(nn.Module):
    def __init__(self, in_dim, out_dim, bottle_neck_dim = 500):
        super(CLS, self).__init__()
        if bottle_neck_dim:
            self.bottleneck = nn.Sequential()
            self.bottleneck.add_module('bottleneck', nn.Linear(in_dim, 64))
            self.bottleneck.add_module('relu', nn.ReLU(inplace = True))
            self.bottleneck.add_module('fc', nn.Linear(64, bottle_neck_dim))
            
            self.fc = nn.Sequential()
            self.fc.add_module('v_linear1', nn.Linear(bottle_neck_dim, 64))
            self.fc.add_module('v_relu1', nn.ReLU(inplace = True))
            self.fc.add_module('v_linear2', nn.Linear(64, out_dim))
            
            self.main = nn.Sequential(
                self.bottleneck,
                self.fc
            )
        else:
            self.main = nn.Sequential()
            self.main.add_module('fc', nn.Linear(in_dim, out_dim))

    def forward(self, x):
        out = [x]
        for module in self.main.children():
            x = module(x)
            out.append(x)
        return out

class AdversarialNetwork(nn.Module):
    def __init__(self):
        super(AdversarialNetwork, self).__init__()
        self.main = nn.Sequential()

    def forward(self, x):
        for module in self.main.children():
            x = module(x)
        return x

class LargeDiscriminator(AdversarialNetwork):
    def __init__(self, in_feature):
        super(LargeDiscriminator, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, 64)
        self.ad_relu1 = nn.ReLU(inplace = True)
        self.ad_layer2 = nn.Linear(64, 128)
        self.ad_relu2 = nn.ReLU(inplace = True)
        self.ad_layer3 = nn.Linear(128, 256)
        self.ad_relu3 = nn.ReLU(inplace = True)
        self.ad_layer4 = nn.Linear(256, 64)
        self.ad_relu4 = nn.ReLU(inplace = True)
        self.ad_layer5 = nn.Linear(64, 1)
        self.ad_sigmoid = nn.Sigmoid()
        self.main = nn.Sequential(
            self.ad_layer1,
            self.ad_relu1,
            self.ad_layer2,
            self.ad_relu2,
            self.ad_layer3,
            self.ad_relu3,
            self.ad_layer4,
            self.ad_relu4,
            self.ad_layer5,
            self.ad_sigmoid
        )
        