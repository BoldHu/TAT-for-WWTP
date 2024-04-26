import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import *

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
        self.ad_layer1 = nn.Linear(in_feature, 128)
        self.ad_relu1 = nn.LeakyReLU(inplace = True)
        self.ad_layer3 = nn.Linear(128, 256)
        self.ad_relu3 = nn.LeakyReLU(inplace = True)
        self.ad_layer4 = nn.Linear(256, 64)
        self.ad_relu4 = nn.LeakyReLU(inplace = True)
        self.ad_layer5 = nn.Linear(64, 1)
        self.ad_sigmoid = nn.Sigmoid()
        self.main = nn.Sequential(
            self.ad_layer1,
            self.ad_relu1,
            self.ad_layer3,
            self.ad_relu3,
            self.ad_layer4,
            self.ad_relu4,
            self.ad_layer5,
            self.ad_sigmoid
        )

class CLS(nn.Module):
    def __init__(self, in_dim, out_dim, bottle_neck_dim = 128):
        super(CLS, self).__init__()
        if bottle_neck_dim:
            self.bottleneck = nn.Sequential()
            self.bottleneck.add_module('bottleneck_linear1', nn.Linear(in_dim, 256))
            self.bottleneck.add_module('bottleneck_relu1', nn.LeakyReLU(inplace = True))
            self.bottleneck.add_module('bottleneck_linear2', nn.Linear(256, 512))
            self.bottleneck.add_module('bottleneck_relu2', nn.LeakyReLU(inplace = True))
            self.bottleneck.add_module('bottleneck_linear3', nn.Linear(512, 512))
            self.bottleneck.add_module('bottleneck_relu3', nn.LeakyReLU(inplace = True))
            self.bottleneck.add_module('bottleneck_linear4', nn.Linear(512, bottle_neck_dim))
            self.bottleneck.add_module('bottleneck_relu4', nn.LeakyReLU(inplace = True))
            self.fc = nn.Sequential()
            self.fc.add_module('fc_linear1', nn.Linear(bottle_neck_dim, 256))
            self.fc.add_module('fc_relu1', nn.LeakyReLU(inplace = True))
            # self.fc.add_module('fc_batchnorm1', nn.BatchNorm1d(256))
            # self.fc.add_module('fc_dropout1', nn.Dropout(p = 0.5))
            self.fc.add_module('fc_linear2', nn.Linear(256, 64))
            self.fc.add_module('fc_relu2', nn.LeakyReLU(inplace = True))
            # self.fc.add_module('fc_batchnorm2', nn.BatchNorm1d(64))
            # self.fc.add_module('fc_dropout2', nn.Dropout(p = 0.5))
            self.output = nn.Sequential()
            self.output.add_module('output_linear', nn.Linear(64, out_dim))
            self.main = nn.Sequential(
                self.bottleneck,
                self.fc,
                self.output
                )
        else:
            self.fc = nn.Linear(in_dim, out_dim)
            self.main = nn.Sequential(
                self.fc,
                nn.Softmax(dim = -1)
            )

    def forward(self, x):
        out = [x]
        for module in self.main.children():
            x = module(x)
            out.append(x)
        return out

class FeatureExtractor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeatureExtractor, self).__init__()
        self.fc = nn.Sequential()
        self.fc.add_module('fc_linear1', nn.Linear(in_dim, 32))
        self.fc.add_module('fc_relu1', nn.LeakyReLU(inplace = True))
        self.fc.add_module('fc_linear2', nn.Linear(32, 64))
        self.fc.add_module('fc_relu2', nn.LeakyReLU(inplace = True))
        self.fc.add_module('fc_linear3', nn.Linear(64, out_dim))
        self.fc.add_module('fc_relu3', nn.LeakyReLU(inplace = True))
        
        # fix the parameters of the feature extractor
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        result = self.fc(x)
        return result
        