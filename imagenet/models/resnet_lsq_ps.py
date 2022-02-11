import torch
import torch.nn as nn
import torchvision.transforms as transforms
import math
from .lsq import Conv2dLSQ, LinearLSQ, ActLSQ, PartialSumLSQ, PartialSumNoQ

__all__ = ['resnet_lsq_ps']

def splitConv1x1(in_planes, out_planes, stride=1, nbits=3):
    "3x3 convolution with padding"
    return Conv2dLSQ(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False, nbits=nbits)

def conv3x3(in_planes, out_planes, stride=1, nbits=3):
    "3x3 convolution with padding"
    return Conv2dLSQ(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, nbits=nbits)

def QuantizePartial(psumq_bits):
    return PartialSumLSQ(psumq_bits=psumq_bits)

def init_model(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

def split_tensor_128(xp):
    x1 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2)
    x2 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2)
    x3 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2)
    x4 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2)
    x5 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2)
    x6 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2)
    x7 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2)
    x8 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2)
    x9 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2)

    return x1,x2,x3,x4,x5,x6,x7,x8,x9

def split_tesnsor_256(xp,max_size = 128):
    x1 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,0,max_size)
    x2 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,0,max_size)
    x3 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,0,max_size)
    x4 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,0,max_size)
    x5 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,0,max_size)
    x6 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,0,max_size)
    x7 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,0,max_size)
    x8 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,0,max_size)
    x9 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,0,max_size)
    x12 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x22 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x32 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x42 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x52 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x62 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x72 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x82 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x92 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size,max_size)

    return x1, x2, x3, x4, x5, x6, x7, x8, x9, x12, x22, x32, x42, x52, x62, x72, x82, x92

def split_tesnsor_384(xp,max_size = 128):
    x1 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,0,max_size)
    x2 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,0,max_size)
    x3 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,0,max_size)
    x4 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,0,max_size)
    x5 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,0,max_size)
    x6 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,0,max_size)
    x7 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,0,max_size)
    x8 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,0,max_size)
    x9 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,0,max_size)
    x12 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x22 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x32 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x42 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x52 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x62 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x72 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x82 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x92 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x13 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x23 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x33 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x43 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x53 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x63 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x73 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x83 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x93 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size*2,max_size)

    return x1, x2, x3, x4, x5, x6, x7, x8, x9, x12, x22, x32, x42, x52, x62, x72, x82, x92, x13, x23, x33, x43, x53, x63, x73, x83, x93

def split_tesnsor_512(xp,max_size = 128):
    x1 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,0,max_size)
    x2 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,0,max_size)
    x3 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,0,max_size)
    x4 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,0,max_size)
    x5 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,0,max_size)
    x6 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,0,max_size)
    x7 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,0,max_size)
    x8 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,0,max_size)
    x9 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,0,max_size)
    x12 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x22 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x32 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x42 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x52 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x62 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x72 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x82 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x92 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size,max_size)
    x13 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x23 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x33 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x43 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x53 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x63 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x73 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x83 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x93 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1,max_size*2,max_size)
    x14 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1, max_size * 3, max_size)
    x24 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1, max_size * 3, max_size)
    x34 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 0, xp.shape[3] - 2).narrow(1, max_size * 3, max_size)
    x44 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1, max_size * 3, max_size)
    x54 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1, max_size * 3, max_size)
    x64 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 1, xp.shape[3] - 2).narrow(1, max_size * 3, max_size)
    x74 = xp.narrow(2, 0, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1, max_size * 3, max_size)
    x84 = xp.narrow(2, 1, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1, max_size * 3, max_size)
    x94 = xp.narrow(2, 2, xp.shape[2] - 2).narrow(3, 2, xp.shape[3] - 2).narrow(1, max_size * 3, max_size)

    return x1, x2, x3, x4, x5, x6, x7, x8, x9, x12, x22, x32, x42, x52, x62, x72, x82, x92, x13, x23, x33, x43, x53, x63, x73, x83, x93, x14, x24, x34, x44, x54, x64, x74, x84, x94

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, nbits=3, psumq_bits=3):
        super(BasicBlock, self).__init__()

        ###########################################     CONV1       ##########################################
        max_size = 128
        groups = math.ceil(inplanes / max_size)

        #padding outside the conv
        self.padding1 = nn.ZeroPad2d(1)
        input_dem = inplanes
        inplanes = min(input_dem,max_size)

        self.conv1 = splitConv1x1(inplanes, planes, stride, nbits=nbits)
        self.psq1 = QuantizePartial(psumq_bits)

        self.conv2 = splitConv1x1(inplanes, planes, stride, nbits=nbits)
        self.psq2 = QuantizePartial(psumq_bits)

        self.conv3 = splitConv1x1(inplanes, planes, stride, nbits=nbits)
        self.psq3 = QuantizePartial(psumq_bits)

        self.conv4 = splitConv1x1(inplanes, planes, stride, nbits=nbits)
        self.psq4 = QuantizePartial(psumq_bits)

        self.conv5 = splitConv1x1(inplanes, planes, stride, nbits=nbits)
        self.psq5 = QuantizePartial(psumq_bits)

        self.conv6 = splitConv1x1(inplanes, planes, stride, nbits=nbits)
        self.psq6 = QuantizePartial(psumq_bits)

        self.conv7 = splitConv1x1(inplanes, planes, stride, nbits=nbits)
        self.psq7 = QuantizePartial(psumq_bits)

        self.conv8 = splitConv1x1(inplanes, planes, stride, nbits=nbits)
        self.psq8 = QuantizePartial(psumq_bits)

        self.conv9 = splitConv1x1(inplanes, planes, stride, nbits=nbits)
        self.psq9 = QuantizePartial(psumq_bits)

        if (input_dem > 128):     #Input channels = 256
            self.conv12 = splitConv1x1(inplanes, planes, stride, nbits=nbits)
            self.psq12 = QuantizePartial(psumq_bits)

            self.conv22 = splitConv1x1(inplanes, planes, stride, nbits=nbits)
            self.psq22 = QuantizePartial(psumq_bits)

            self.conv32 = splitConv1x1(inplanes, planes, stride, nbits=nbits)
            self.psq32 = QuantizePartial(psumq_bits)

            self.conv42 = splitConv1x1(inplanes, planes, stride, nbits=nbits)
            self.psq42 = QuantizePartial(psumq_bits)

            self.conv52 = splitConv1x1(inplanes, planes, stride, nbits=nbits)
            self.psq52 = QuantizePartial(psumq_bits)

            self.conv62 = splitConv1x1(inplanes, planes, stride, nbits=nbits)
            self.psq62 = QuantizePartial(psumq_bits)

            self.conv72 = splitConv1x1(inplanes, planes, stride, nbits=nbits)
            self.psq72 = QuantizePartial(psumq_bits)

            self.conv82 = splitConv1x1(inplanes, planes, stride, nbits=nbits)
            self.psq82 = QuantizePartial(psumq_bits)

            self.conv92 = splitConv1x1(inplanes, planes, stride, nbits=nbits)
            self.psq92 = QuantizePartial(psumq_bits)


        if (input_dem > 256):       #Input channels = 384
            self.conv13 = splitConv1x1(inplanes, planes, stride, nbits=nbits)
            self.psq13 = QuantizePartial(psumq_bits)

            self.conv23 = splitConv1x1(inplanes, planes, stride, nbits=nbits)
            self.psq23 = QuantizePartial(psumq_bits)

            self.conv33 = splitConv1x1(inplanes, planes, stride, nbits=nbits)
            self.psq33 = QuantizePartial(psumq_bits)

            self.conv43 = splitConv1x1(inplanes, planes, stride, nbits=nbits)
            self.psq43 = QuantizePartial(psumq_bits)

            self.conv53 = splitConv1x1(inplanes, planes, stride, nbits=nbits)
            self.psq53 = QuantizePartial(psumq_bits)

            self.conv63 = splitConv1x1(inplanes, planes, stride, nbits=nbits)
            self.psq63 = QuantizePartial(psumq_bits)

            self.conv73 = splitConv1x1(inplanes, planes, stride, nbits=nbits)
            self.psq73 = QuantizePartial(psumq_bits)

            self.conv83 = splitConv1x1(inplanes, planes, stride, nbits=nbits)
            self.psq83 = QuantizePartial(psumq_bits)

            self.conv93 = splitConv1x1(inplanes, planes, stride, nbits=nbits)
            self.psq93 = QuantizePartial(psumq_bits)


        if (input_dem > 384):       #Input channels = 512
            self.conv14 = splitConv1x1(inplanes, planes, stride, nbits=nbits)
            self.psq14 = QuantizePartial(psumq_bits)

            self.conv24 = splitConv1x1(inplanes, planes, stride, nbits=nbits)
            self.psq24 = QuantizePartial(psumq_bits)

            self.conv34 = splitConv1x1(inplanes, planes, stride, nbits=nbits)
            self.psq34 = QuantizePartial(psumq_bits)

            self.conv44 = splitConv1x1(inplanes, planes, stride, nbits=nbits)
            self.psq44 = QuantizePartial(psumq_bits)

            self.conv54 = splitConv1x1(inplanes, planes, stride, nbits=nbits)
            self.psq54 = QuantizePartial(psumq_bits)

            self.conv64 = splitConv1x1(inplanes, planes, stride, nbits=nbits)
            self.psq64 = QuantizePartial(psumq_bits)

            self.conv74 = splitConv1x1(inplanes, planes, stride, nbits=nbits)
            self.psq74 = QuantizePartial(psumq_bits)

            self.conv84 = splitConv1x1(inplanes, planes, stride, nbits=nbits)
            self.psq84 = QuantizePartial(psumq_bits)

            self.conv94 = splitConv1x1(inplanes, planes, stride, nbits=nbits)
            self.psq94 = QuantizePartial(psumq_bits)


        ###########################################     END     ##########################################

        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.Sequential(nn.ReLU(inplace=True), ActLSQ(nbits=nbits))

        ###########################################     CONV2       ##########################################

        inplanes = min(max_size,planes)
        groups = math.ceil(planes / max_size)
        self.padding2 = nn.ZeroPad2d(1)
        self.conv1_2 = splitConv1x1(inplanes, planes, nbits=nbits)
        self.psq1_2 = QuantizePartial(psumq_bits)

        self.conv2_2 = splitConv1x1(inplanes, planes, nbits=nbits)
        self.psq2_2 = QuantizePartial(psumq_bits)

        self.conv3_2 = splitConv1x1(inplanes, planes, nbits=nbits)
        self.psq3_2 = QuantizePartial(psumq_bits)

        self.conv4_2 = splitConv1x1(inplanes, planes, nbits=nbits)
        self.psq4_2 = QuantizePartial(psumq_bits)

        self.conv5_2 = splitConv1x1(inplanes, planes, nbits=nbits)
        self.psq5_2 = QuantizePartial(psumq_bits)

        self.conv6_2 = splitConv1x1(inplanes, planes, nbits=nbits)
        self.psq6_2 = QuantizePartial(psumq_bits)

        self.conv7_2 = splitConv1x1(inplanes, planes, nbits=nbits)
        self.psq7_2 = QuantizePartial(psumq_bits)

        self.conv8_2 = splitConv1x1(inplanes, planes, nbits=nbits)
        self.psq8_2 = QuantizePartial(psumq_bits)

        self.conv9_2 = splitConv1x1(inplanes, planes, nbits=nbits)
        self.psq9_2 = QuantizePartial(psumq_bits)

        if(planes>128):
            self.conv12_2 = splitConv1x1(inplanes, planes, nbits=nbits)
            self.psq12_2 = QuantizePartial(psumq_bits)

            self.conv22_2 = splitConv1x1(inplanes, planes, nbits=nbits)
            self.psq22_2 = QuantizePartial(psumq_bits)

            self.conv32_2 = splitConv1x1(inplanes, planes, nbits=nbits)
            self.psq32_2 = QuantizePartial(psumq_bits)

            self.conv42_2 = splitConv1x1(inplanes, planes, nbits=nbits)
            self.psq42_2 = QuantizePartial(psumq_bits)

            self.conv52_2 = splitConv1x1(inplanes, planes, nbits=nbits)
            self.psq52_2 = QuantizePartial(psumq_bits)

            self.conv62_2 = splitConv1x1(inplanes, planes, nbits=nbits)
            self.psq62_2 = QuantizePartial(psumq_bits)

            self.conv72_2 = splitConv1x1(inplanes, planes, nbits=nbits)
            self.psq72_2 = QuantizePartial(psumq_bits)

            self.conv82_2 = splitConv1x1(inplanes, planes, nbits=nbits)
            self.psq82_2 = QuantizePartial(psumq_bits)

            self.conv92_2 = splitConv1x1(inplanes, planes, nbits=nbits)
            self.psq92_2 = QuantizePartial(psumq_bits)


        if (planes > 256):
            self.conv13_2 = splitConv1x1(inplanes, planes, nbits=nbits)
            self.psq13_2 = QuantizePartial(psumq_bits)

            self.conv23_2 = splitConv1x1(inplanes, planes, nbits=nbits)
            self.psq23_2 = QuantizePartial(psumq_bits)

            self.conv33_2 = splitConv1x1(inplanes, planes, nbits=nbits)
            self.psq33_2 = QuantizePartial(psumq_bits)

            self.conv43_2 = splitConv1x1(inplanes, planes, nbits=nbits)
            self.psq43_2 = QuantizePartial(psumq_bits)

            self.conv53_2 = splitConv1x1(inplanes, planes, nbits=nbits)
            self.psq53_2 = QuantizePartial(psumq_bits)

            self.conv63_2 = splitConv1x1(inplanes, planes, nbits=nbits)
            self.psq63_2 = QuantizePartial(psumq_bits)

            self.conv73_2 = splitConv1x1(inplanes, planes, nbits=nbits)
            self.psq73_2 = QuantizePartial(psumq_bits)

            self.conv83_2 = splitConv1x1(inplanes, planes, nbits=nbits)
            self.psq83_2 = QuantizePartial(psumq_bits)

            self.conv93_2 = splitConv1x1(inplanes, planes, nbits=nbits)
            self.psq93_2 = QuantizePartial(psumq_bits)


        if (planes > 384):
            self.conv14_2 = splitConv1x1(inplanes, planes, nbits=nbits)
            self.psq14_2 = QuantizePartial(psumq_bits)

            self.conv24_2 = splitConv1x1(inplanes, planes, nbits=nbits)
            self.psq24_2 = QuantizePartial(psumq_bits)

            self.conv34_2 = splitConv1x1(inplanes, planes, nbits=nbits)
            self.psq34_2 = QuantizePartial(psumq_bits)

            self.conv44_2 = splitConv1x1(inplanes, planes, nbits=nbits)
            self.psq44_2 = QuantizePartial(psumq_bits)

            self.conv54_2 = splitConv1x1(inplanes, planes, nbits=nbits)
            self.psq54_2 = QuantizePartial(psumq_bits)

            self.conv64_2 = splitConv1x1(inplanes, planes, nbits=nbits)
            self.psq64_2 = QuantizePartial(psumq_bits)

            self.conv74_2 = splitConv1x1(inplanes, planes, nbits=nbits)
            self.psq74_2 = QuantizePartial(psumq_bits)

            self.conv84_2 = splitConv1x1(inplanes, planes, nbits=nbits)
            self.psq84_2 = QuantizePartial(psumq_bits)

            self.conv94_2 = splitConv1x1(inplanes, planes, nbits=nbits)
            self.psq94_2 = QuantizePartial(psumq_bits)


        ###########################################     END     ##########################################

        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        inplanes = x.shape[1]
        max_size = 128
        groups = math.ceil(inplanes / max_size)

        residual = x.clone()
        xp = x
        xp = self.padding1(xp)

        #splitting x
        if(xp.shape[1]<=128):
            x1,x2,x3,x4,x5,x6,x7,x8,x9 = split_tensor_128(xp)
        elif(xp.shape[1]==256):
            x1,x2,x3,x4,x5,x6,x7,x8,x9,x12,x22,x32,x42,x52,x62,x72,x82,x92= split_tesnsor_256(xp)
        elif (xp.shape[1] == 384):
            x1,x2,x3,x4,x5,x6,x7,x8,x9,x12,x22,x32,x42,x52,x62,x72,x82,x92,x13,x23,x33,x43,x53,x63,x73,x83,x93 = split_tesnsor_384(xp)
        elif (xp.shape[1] == 512):
            x1,x2,x3,x4,x5,x6,x7,x8,x9,x12,x22,x32,x42,x52,x62,x72,x82,x92,x13,x23,x33,x43,x53,x63,x73,x83,x93,x14,x24,x34,x44,x54,x64,x74,x84,x94 = split_tesnsor_512(xp)
        else:
            print(xp.shape)
            print("============ILLEGAL INPUT======================")

        out = []

        #Perform conv and quantize partial sums
        out.append(self.psq1(self.conv1(x1)))
        out.append(self.psq2(self.conv2(x2)))
        out.append(self.psq3(self.conv3(x3)))
        out.append(self.psq4(self.conv4(x4)))
        out.append(self.psq5(self.conv5(x5)))
        out.append(self.psq6(self.conv6(x6)))
        out.append(self.psq7(self.conv7(x7)))
        out.append(self.psq8(self.conv8(x8)))
        out.append(self.psq9(self.conv9(x9)))

        #out = self.conv1(x)

        if (xp.shape[1]>128):

            out.append(self.psq12(self.conv12(x12)))
            out.append(self.psq22(self.conv22(x22)))
            out.append(self.psq32(self.conv32(x32)))
            out.append(self.psq42(self.conv42(x42)))
            out.append(self.psq52(self.conv52(x52)))
            out.append(self.psq62(self.conv62(x62)))
            out.append(self.psq72(self.conv72(x72)))
            out.append(self.psq82(self.conv82(x82)))
            out.append(self.psq92(self.conv92(x92)))

        if (xp.shape[1] > 256):
            out.append(self.psq13(self.conv13(x13)))
            out.append(self.psq23(self.conv23(x23)))
            out.append(self.psq33(self.conv33(x33)))
            out.append(self.psq43(self.conv43(x43)))
            out.append(self.psq53(self.conv53(x53)))
            out.append(self.psq63(self.conv63(x63)))
            out.append(self.psq73(self.conv73(x73)))
            out.append(self.psq83(self.conv83(x83)))
            out.append(self.psq93(self.conv93(x93)))

        if (xp.shape[1] > 384):
            out.append(self.psq14(self.conv14(x14)))
            out.append(self.psq24(self.conv24(x24)))
            out.append(self.psq34(self.conv34(x34)))
            out.append(self.psq44(self.conv44(x44)))
            out.append(self.psq54(self.conv54(x54)))
            out.append(self.psq64(self.conv64(x64)))
            out.append(self.psq74(self.conv74(x74)))
            out.append(self.psq84(self.conv84(x84)))
            out.append(self.psq94(self.conv94(x94)))

        output = torch.zeros(out[0].shape).cuda()

        #accumulate all partial sums
        for out_tensor in out:
            output = output + out_tensor

        output = self.bn1(output)
        xn = self.relu(output)

        xn = self.padding2(xn)
        inplanes = x.shape[1]
        groups = math.ceil(inplanes / max_size)

        if (xn.shape[1] <= 128):
            x1, x2, x3, x4, x5, x6, x7, x8, x9 = split_tensor_128(xn)
        elif (xn.shape[1] == 256):
            x1, x2, x3, x4, x5, x6, x7, x8, x9, x12, x22, x32, x42, x52, x62, x72, x82, x92 = split_tesnsor_256(xn)
        elif (xn.shape[1] == 384):
            x1, x2, x3, x4, x5, x6, x7, x8, x9, x12, x22, x32, x42, x52, x62, x72, x82, x92, x13, x23, x33, x43, x53, x63, x73, x83, x93 = split_tesnsor_384(
                xn)
        elif (xn.shape[1] == 512):
            x1, x2, x3, x4, x5, x6, x7, x8, x9, x12, x22, x32, x42, x52, x62, x72, x82, x92, x13, x23, x33, x43, x53, x63, x73, x83, x93, x14, x24, x34, x44, x54, x64, x74, x84, x94 = split_tesnsor_512(
                xn)
        else:
            print(xn.shape)
            print("============ILLEGAL INPUT======================")

        out = []

        out.append(self.psq1_2(self.conv1_2(x1)))
        out.append(self.psq2_2(self.conv2_2(x2)))
        out.append(self.psq3_2(self.conv3_2(x3)))
        out.append(self.psq4_2(self.conv4_2(x4)))
        out.append(self.psq5_2(self.conv5_2(x5)))
        out.append(self.psq6_2(self.conv6_2(x6)))
        out.append(self.psq7_2(self.conv7_2(x7)))
        out.append(self.psq8_2(self.conv8_2(x8)))
        out.append(self.psq9_2(self.conv9_2(x9)))

        if(xn.shape[1]>128):
            out.append(self.psq12_2(self.conv12_2(x12)))
            out.append(self.psq22_2(self.conv22_2(x22)))
            out.append(self.psq32_2(self.conv32_2(x32)))
            out.append(self.psq42_2(self.conv42_2(x42)))
            out.append(self.psq52_2(self.conv52_2(x52)))
            out.append(self.psq62_2(self.conv62_2(x62)))
            out.append(self.psq72_2(self.conv72_2(x72)))
            out.append(self.psq82_2(self.conv82_2(x82)))
            out.append(self.psq92_2(self.conv92_2(x92)))

        if (xn.shape[1] > 256):
            out.append(self.psq13_2(self.conv13_2(x13)))
            out.append(self.psq23_2(self.conv23_2(x23)))
            out.append(self.psq33_2(self.conv33_2(x33)))
            out.append(self.psq43_2(self.conv43_2(x43)))
            out.append(self.psq53_2(self.conv53_2(x53)))
            out.append(self.psq63_2(self.conv63_2(x63)))
            out.append(self.psq73_2(self.conv73_2(x73)))
            out.append(self.psq83_2(self.conv83_2(x83)))
            out.append(self.psq93_2(self.conv93_2(x93)))

        if (xn.shape[1] > 384):
            out.append(self.psq14_2(self.conv14_2(x14)))
            out.append(self.psq24_2(self.conv24_2(x24)))
            out.append(self.psq34_2(self.conv34_2(x34)))
            out.append(self.psq44_2(self.conv44_2(x44)))
            out.append(self.psq54_2(self.conv54_2(x54)))
            out.append(self.psq64_2(self.conv64_2(x64)))
            out.append(self.psq74_2(self.conv74_2(x74)))
            out.append(self.psq84_2(self.conv84_2(x84)))
            out.append(self.psq94_2(self.conv94_2(x94)))

        output = torch.zeros(out[0].shape).cuda()

        for out_tensor in out:
            output = output + out_tensor

        out = self.bn2(output)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

    def _make_layer(self, block, planes, blocks, stride=1, nbits=3, psumq_bits=3):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2dLSQ(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False, nbits=nbits),
                nn.BatchNorm2d(planes * block.expansion),
                ActLSQ(nbits=nbits)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            nbits=nbits, psumq_bits=psumq_bits))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, nbits=nbits, psumq_bits=psumq_bits))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNet_imagenet(ResNet):

    def __init__(self, num_classes=1000,
                 block=BasicBlock, layers=[3, 4, 23, 3], nbits=3, psumq_bits=3):
        super(ResNet_imagenet, self).__init__()
        self.inplanes = 64
        #first layer is not quantized
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], nbits=nbits, psumq_bits=psumq_bits)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, nbits=nbits, psumq_bits=psumq_bits)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, nbits=nbits, psumq_bits=psumq_bits)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, nbits=nbits, psumq_bits=psumq_bits)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        init_model(self)
        self.regime = {
            0: {'optimizer': 'SGD', 'lr': 1e-3,
                'weight_decay': 1e-4, 'momentum': 0.9},
            30: {'lr': 1e-4},
            60: {'lr': 1e-5, 'weight_decay': 0},
            90: {'lr': 1e-6}
        }


def resnet_lsq_ps(**kwargs):
    nbits = kwargs.get('nbits', 3)
    psumq_bits = kwargs.get('psumq_bits', 3)

    #resnet18
    return ResNet_imagenet(num_classes=1000, block=BasicBlock,
                           layers=[2, 2, 2, 2], nbits=nbits, psumq_bits=psumq_bits)
