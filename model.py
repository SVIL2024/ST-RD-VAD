import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
import math
from memory import Memory

class ConvBlock(nn.Module):
    def __init__(self, in_channel, kernel_size, filters, stride):
        super(ConvBlock, self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv3d(in_channel, F1, 1, stride=stride, padding=0, bias=False),
            nn.BatchNorm3d(F1),
            nn.ReLU(True),
            nn.Conv3d(F1, F2, kernel_size, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(F2),
            nn.ReLU(True),
            nn.Conv3d(F2, F3, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(F3)
        )
        self.shortcut_1 = nn.Conv3d(in_channel, F3, 1, stride=stride, padding=0, bias=False)
        self.batch_1 = nn.BatchNorm3d(F3)
        self.relu_1 = nn.ReLU(inplace=True)

    def forward(self, X):
        X_shortcut = self.shortcut_1(X)
        X_shortcut = self.batch_1(X_shortcut)
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X


class IdentityBlock(nn.Module):
    def __init__(self, in_channel, kernel_size, filters):
        super(IdentityBlock, self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv3d(in_channel, F1, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(F1),
            nn.ReLU(True),
            nn.Conv3d(F1, F2, kernel_size, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(F2),
            nn.ReLU(True),
            nn.Conv3d(F2, F3, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(F3)
        )
        self.relu_1 = nn.ReLU(True)

    def forward(self, X):
        X_shortcut = X
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X


class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channel, kernel_size, filters):
        super(ConvTransposeBlock, self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.ConvTranspose3d(in_channel, F1, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm3d(F1),
            nn.ReLU(True),
            nn.Conv3d(F1, F2, kernel_size, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(F2),
            nn.ReLU(True),
            nn.Conv3d(F2, F3, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(F3),
        )
        self.shortcut_1 = nn.ConvTranspose3d(in_channel, F3, kernel_size=2, stride=2, padding=0, bias=False)
        self.batch_1 = nn.BatchNorm3d(F3)
        self.relu_1 = nn.ReLU(inplace=True)

    def forward(self, X):
        X_shortcut = self.shortcut_1(X)
        X_shortcut = self.batch_1(X_shortcut)
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X


class OCBE(nn.Module):
    def __init__(self, in_channels_list=[480, 832, 1024]):
        super(OCBE, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels_list[0], 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(True),
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels_list[1], 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(True),
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(True),
        )

        self.branch3 = nn.Sequential(
            nn.ConvTranspose3d(in_channels_list[2], 512, kernel_size=2, stride=2),
            nn.BatchNorm3d(512),
            nn.ReLU(True),
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(True),
        )

        self.fusion = nn.Sequential(
            nn.Conv3d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(512),
            nn.ReLU(True),
        )

        self.resblock = nn.Sequential(
            ConvBlock(in_channel=512, kernel_size=3, filters=[256, 256, 1024], stride=2),
            IdentityBlock(in_channel=1024, kernel_size=3, filters=[256, 256, 1024]),
            IdentityBlock(in_channel=1024, kernel_size=3, filters=[256, 256, 1024]),
        )

    def forward(self, x1, x2, x3):
        b1 = self.branch1(x1)
        b2 = self.branch2(x2)
        b3 = self.branch3(x3)

        output = torch.cat((b1, b2, b3), dim=1)
        output = self.fusion(output)
        output = self.resblock(output)
        return output


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.layer3 = nn.Sequential(
            ConvTransposeBlock(in_channel=1024, kernel_size=3, filters=[256, 512, 1024]),
            IdentityBlock(in_channel=1024, kernel_size=3, filters=[256, 512, 1024]),
        )

        self.layer3_add = nn.Sequential(
            nn.ConvTranspose3d(1024, 512, kernel_size=(3, 3, 3), stride=(2, 2, 2),
                               padding=(1, 1, 1), output_padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(True),
        )

        self.layer1 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(True),
        )

        self.match_feature1 = nn.Conv3d(512, 480, kernel_size=1)
        self.match_feature2 = nn.Conv3d(512, 832, kernel_size=1)
        self.match_feature3 = nn.Conv3d(512, 1024, kernel_size=1)

    def forward(self, x):
        x = self.layer3(x)
        x = self.layer3_add(x)

        feature1 = self.match_feature1(x)

        x = self.layer2(x)
        feature2 = self.match_feature2(x)

        x = self.layer1(x)
        feature3 = self.match_feature3(x)

        return feature1, feature2, feature3


class OcbeAndDecoder(nn.Module):
    def __init__(self, in_channels_list=[480, 832, 1024], mem_dim=100, shrink_thres=0.0025):
        super(OcbeAndDecoder, self).__init__()
        self.ocbe = OCBE(in_channels_list=in_channels_list)
        self.memory = Memory(mem_dim=mem_dim, fea_dim=1024, shrink_thres=shrink_thres)
        self.decoder = Decoder()

    def forward(self, e_feature1, e_feature2, e_feature3):
        ocbe_output = self.ocbe(e_feature1, e_feature2, e_feature3)  # (B, 1024, 2, 8, 8)
        B, C, D, H, W = ocbe_output.shape

        z_flat = ocbe_output.permute(0, 2, 3, 4, 1).reshape(-1, C)

        mem_out = self.memory(z_flat)
        z_hat = mem_out['out']
        attention = mem_out['att_weight']

        z_hat = z_hat.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)

        decoded_feature1, decoded_feature2, decoded_feature3 = self.decoder(z_hat)
        return decoded_feature1, decoded_feature2, decoded_feature3, attention