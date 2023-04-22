import torch
import torch.nn as nn
from module.backbone.ResNet import ResNet
import torchvision.models as models
import torch.nn.functional as F
from module.cbam import ChannelGate, SpatialGate
from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class MCST(nn.Module):
    def __init__(self, in_channel, out_channel, img_size, num_heads):
        super(MCST, self).__init__()
        self.relu = nn.ReLU(True)
        self.branchRB = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )

        self.branchHDC = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=2, dilation=2),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5),
        )

        self.branchST = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            SwinTransformerSys(self, img_size, in_chans=out_channel, embed_dim=out_channel, num_classes=out_channel, depths=[2],
                               num_heads=num_heads),
            # BasicConv2d(in_channel, out_channel, 1),
        )

        self.conv_cat = BasicConv2d(3 * out_channel, out_channel, 3, padding=1)
        self.conv1x1 = BasicConv2d(in_channel, out_channel, 1)
        self.conv3x3 = BasicConv2d(in_channel, out_channel, 3)

    def  forward(self, x):
        rb = self.branchRB(x)
        hdc = self.branchHDC(x)
        st = self.branchST(x)
        x_cat = self.conv_cat(torch.cat((rb, hdc, st), 1))
        x = self.conv3x3(x_cat)
        x = self.relu(x)
        return x

class MAF(nn.Module):
    def __init__(self, channels_high, channels_low):
        super(MAF, self).__init__()
        self.deConv = nn.ConvTranspose2d(channels_high, channels_high, 2, stride=2)

        self.ChannelGate = ChannelGate(channels_high)
        self.SpatialGate = SpatialGate()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv1x1(channels_high, channels_low)
        self.conv2 = conv3x3(channels_high, channels_low)


    def forward(self, fms_high, fms_low):
        x1 = self.deConv(fms_high)

        x2 = self.ChannelGate(fms_high)
        x2 = self.deConv(x2)
        x2 = self.SpatialGate(x2)
        x3 = self.conv2(fms_low)
        x2 = x2 * x3

        out = x1 + x2
        out = self.relu(out)

        return out


class ECA(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)




class mcst_unet(nn.Module):
    def __init__(self, n_class=1):
        super(mcst_unet, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet = ResNet()
        # ---- Receptive Field Block like module ----

        self.mcst1 = MCST(256, 128, 96, [3])
        self.mcst2 = MCST(512, 256, 48, [6])
        self.mcst3 = MCST(1024, 512, 24, [12])
        self.mcst4 = MCST(2048, 1024, 12, [24])

        bottom_ch = 1024
        self.maf3 = MAF(bottom_ch, 512)
        self.maf2 = MAF(bottom_ch // 2, 256)
        self.maf1 = MAF(bottom_ch // 4, 128)

        self.conv1_1 = conv1x1(128, 1)

        if self.training:
            self.initialize_weights()
            print('initialize_weights')


    def forward(self, x):
        x = self.resnet.conv1(x)        # 64, 192, 192
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)

        # ---- low-level features ----
        x = self.resnet.maxpool(x)      # bs, 64, 96, 96
        x1 = self.resnet.layer1(x)      # bs, 256, 96, 96

        # ---- high-level features ----
        x2 = self.resnet.layer2(x1)     # bs, 512, 48, 48

        x3 = self.resnet.layer3(x2)     # bs, 1024, 24, 24

        x4 = self.resnet.layer4(x3)     # bs, 2048, 12, 12

        x1_mcst = self.mcst1(x1)        # 256 -> 128
        x2_mcst = self.mcst2(x2)        # 512 -> 256
        x3_mcst = self.mcst3(x3)        # 1024 -> 512
        x4_mcst = self.mcst4(x4)        # 2048 -> 1024

        x3 = self.maf3(x4_mcst, x3_mcst)  # 1/16
        x2 = self.maf2(x3, x2_mcst)  # 1/8
        x1 = self.maf1(x2, x1_mcst)  # 1/4

        map_1 = self.conv1_1(x1)

        lateral_map_1 = F.interpolate(map_1, scale_factor=4, mode='bilinear')

        return lateral_map_1

    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True) #If True, the pre-trained resnet50 will be loaded.
        pretrained_dict = res50.state_dict()
        model_dict = self.resnet.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.resnet.load_state_dict(model_dict)


