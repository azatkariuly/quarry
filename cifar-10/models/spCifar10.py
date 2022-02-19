import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Function
from .binarized_modules import  BinarizeLinear, BinarizeConv2d

# True by default
bias_bool = False

def splitConv1x1(in_planes, out_planes, stride=1):
    return BinarizeConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias_bool)

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


class VGG_Cifar10(nn.Module):

    def __init__(self, num_classes=1000):
        super(VGG_Cifar10, self).__init__()

        #skip first layer
        self.conv1 = BinarizeConv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.batchNorm1 = nn.BatchNorm2d(128)
        self.hardTanh1 = nn.Hardtanh(inplace=True)

        #padding outside the conv
        self.padding = nn.ZeroPad2d(1)

        #split to 9
        #self.conv2 = BinarizeConv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.conv21 = splitConv1x1(128, 128)
        self.conv22 = splitConv1x1(128, 128)
        self.conv23 = splitConv1x1(128, 128)
        self.conv24 = splitConv1x1(128, 128)
        self.conv25 = splitConv1x1(128, 128)
        self.conv26 = splitConv1x1(128, 128)
        self.conv27 = splitConv1x1(128, 128)
        self.conv28 = splitConv1x1(128, 128)
        self.conv29 = splitConv1x1(128, 128)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batchNorm2 = nn.BatchNorm2d(128)
        self.hardTanh2 = nn.Hardtanh(inplace=True)

        #split to 9
        #self.conv3 = BinarizeConv2d(128, 256, kernel_size=3, padding=1, bias=True)
        self.conv31 = splitConv1x1(128, 256)
        self.conv32 = splitConv1x1(128, 256)
        self.conv33 = splitConv1x1(128, 256)
        self.conv34 = splitConv1x1(128, 256)
        self.conv35 = splitConv1x1(128, 256)
        self.conv36 = splitConv1x1(128, 256)
        self.conv37 = splitConv1x1(128, 256)
        self.conv38 = splitConv1x1(128, 256)
        self.conv39 = splitConv1x1(128, 256)

        self.batchNorm3 = nn.BatchNorm2d(256)
        self.hardTanh3 = nn.Hardtanh(inplace=True)

        #split to 18
        #self.conv4 = BinarizeConv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.conv41 = splitConv1x1(128, 256)
        self.conv42 = splitConv1x1(128, 256)
        self.conv43 = splitConv1x1(128, 256)
        self.conv44 = splitConv1x1(128, 256)
        self.conv45 = splitConv1x1(128, 256)
        self.conv46 = splitConv1x1(128, 256)
        self.conv47 = splitConv1x1(128, 256)
        self.conv48 = splitConv1x1(128, 256)
        self.conv49 = splitConv1x1(128, 256)
        self.conv41_1 = splitConv1x1(128, 256)
        self.conv42_1 = splitConv1x1(128, 256)
        self.conv43_1 = splitConv1x1(128, 256)
        self.conv44_1 = splitConv1x1(128, 256)
        self.conv45_1 = splitConv1x1(128, 256)
        self.conv46_1 = splitConv1x1(128, 256)
        self.conv47_1 = splitConv1x1(128, 256)
        self.conv48_1 = splitConv1x1(128, 256)
        self.conv49_1 = splitConv1x1(128, 256)

        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batchNorm4 = nn.BatchNorm2d(256)
        self.hardTanh4 = nn.Hardtanh(inplace=True)

        #split to 18
        #self.conv5 = BinarizeConv2d(256, 512, kernel_size=3, padding=1, bias=True)
        self.conv51 = splitConv1x1(128, 512)
        self.conv52 = splitConv1x1(128, 512)
        self.conv53 = splitConv1x1(128, 512)
        self.conv54 = splitConv1x1(128, 512)
        self.conv55 = splitConv1x1(128, 512)
        self.conv56 = splitConv1x1(128, 512)
        self.conv57 = splitConv1x1(128, 512)
        self.conv58 = splitConv1x1(128, 512)
        self.conv59 = splitConv1x1(128, 512)
        self.conv51_1 = splitConv1x1(128, 512)
        self.conv52_1 = splitConv1x1(128, 512)
        self.conv53_1 = splitConv1x1(128, 512)
        self.conv54_1 = splitConv1x1(128, 512)
        self.conv55_1 = splitConv1x1(128, 512)
        self.conv56_1 = splitConv1x1(128, 512)
        self.conv57_1 = splitConv1x1(128, 512)
        self.conv58_1 = splitConv1x1(128, 512)
        self.conv59_1 = splitConv1x1(128, 512)

        self.batchNorm5 = nn.BatchNorm2d(512)
        self.hardTanh5 = nn.Hardtanh(inplace=True)

        #split to 36
        #self.conv6 = BinarizeConv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv61 = splitConv1x1(128, 512)
        self.conv62 = splitConv1x1(128, 512)
        self.conv63 = splitConv1x1(128, 512)
        self.conv64 = splitConv1x1(128, 512)
        self.conv65 = splitConv1x1(128, 512)
        self.conv66 = splitConv1x1(128, 512)
        self.conv67 = splitConv1x1(128, 512)
        self.conv68 = splitConv1x1(128, 512)
        self.conv69 = splitConv1x1(128, 512)
        self.conv61_1 = splitConv1x1(128, 512)
        self.conv62_1 = splitConv1x1(128, 512)
        self.conv63_1 = splitConv1x1(128, 512)
        self.conv64_1 = splitConv1x1(128, 512)
        self.conv65_1 = splitConv1x1(128, 512)
        self.conv66_1 = splitConv1x1(128, 512)
        self.conv67_1 = splitConv1x1(128, 512)
        self.conv68_1 = splitConv1x1(128, 512)
        self.conv69_1 = splitConv1x1(128, 512)
        self.conv61_2 = splitConv1x1(128, 512)
        self.conv62_2 = splitConv1x1(128, 512)
        self.conv63_2 = splitConv1x1(128, 512)
        self.conv64_2 = splitConv1x1(128, 512)
        self.conv65_2 = splitConv1x1(128, 512)
        self.conv66_2 = splitConv1x1(128, 512)
        self.conv67_2 = splitConv1x1(128, 512)
        self.conv68_2 = splitConv1x1(128, 512)
        self.conv69_2 = splitConv1x1(128, 512)
        self.conv61_3 = splitConv1x1(128, 512)
        self.conv62_3 = splitConv1x1(128, 512)
        self.conv63_3 = splitConv1x1(128, 512)
        self.conv64_3 = splitConv1x1(128, 512)
        self.conv65_3 = splitConv1x1(128, 512)
        self.conv66_3 = splitConv1x1(128, 512)
        self.conv67_3 = splitConv1x1(128, 512)
        self.conv68_3 = splitConv1x1(128, 512)
        self.conv69_3 = splitConv1x1(128, 512)

        self.maxpool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batchNorm6 = nn.BatchNorm2d(512)
        self.hardTanh6 = nn.Hardtanh(inplace=True)


        self.linear7 = BinarizeLinear(512 * 4 * 4, 1024, bias=True)
        self.batchNorm7 = nn.BatchNorm1d(1024)
        self.hardTanh7 = nn.Hardtanh(inplace=True)

        self.linear8 = BinarizeLinear(1024, 1024, bias=True)
        self.batchNorm8 = nn.BatchNorm1d(1024)
        self.hardTanh8 = nn.Hardtanh(inplace=True)

        self.linear9 = BinarizeLinear(1024, num_classes, bias=True)
        self.batchNorm9 = nn.BatchNorm1d(num_classes, affine=False)

        self.regime = {
            0: {'optimizer': 'Adam', 'betas': (0.9, 0.999),'lr': 1e-2},
            100: {'lr': 1e-3},
            200: {'lr': 1e-4},
            300: {'lr': 1e-5},
            400: {'lr': 1e-6}
        }

    def forward(self, x):
        #skip first layer
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.hardTanh1(x)

        #x = self.conv2(x)
        #split x
        xp = x
        xp = self.padding(xp)
        x1, x2, x3, x4, x5, x6, x7, x8, x9 = split_tensor_128(xp)

        out = []
        out.append(self.conv21(x1))
        out.append(self.conv22(x2))
        out.append(self.conv23(x3))
        out.append(self.conv24(x4))
        out.append(self.conv25(x5))
        out.append(self.conv26(x6))
        out.append(self.conv27(x7))
        out.append(self.conv28(x8))
        out.append(self.conv29(x9))

        output = torch.zeros(out[0].shape).cuda()
        #accumulate all partial sums
        for out_tensor in out:
            output = output + out_tensor

        x = self.maxpool2(output)
        x = self.batchNorm2(x)
        x = self.hardTanh2(x)


        #x = self.conv3(x)
        #split x
        xp = x
        xp = self.padding(xp)
        x1, x2, x3, x4, x5, x6, x7, x8, x9 = split_tensor_128(xp)

        out = []
        out.append(self.conv31(x1))
        out.append(self.conv32(x2))
        out.append(self.conv33(x3))
        out.append(self.conv34(x4))
        out.append(self.conv35(x5))
        out.append(self.conv36(x6))
        out.append(self.conv37(x7))
        out.append(self.conv38(x8))
        out.append(self.conv39(x9))

        output = torch.zeros(out[0].shape).cuda()
        #accumulate all partial sums
        for out_tensor in out:
            output = output + out_tensor

        x = self.batchNorm3(output)
        x = self.hardTanh3(x)

        #x = self.conv4(x)
        #split x
        xp = x
        xp = self.padding(xp)
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x12, x22, x32, x42, x52, x62, x72, x82, x92 = split_tesnsor_256(xp)

        out = []
        out.append(self.conv41(x1))
        out.append(self.conv42(x2))
        out.append(self.conv43(x3))
        out.append(self.conv44(x4))
        out.append(self.conv45(x5))
        out.append(self.conv46(x6))
        out.append(self.conv47(x7))
        out.append(self.conv48(x8))
        out.append(self.conv49(x9))
        out.append(self.conv41_1(x12))
        out.append(self.conv42_1(x22))
        out.append(self.conv43_1(x32))
        out.append(self.conv44_1(x42))
        out.append(self.conv45_1(x52))
        out.append(self.conv46_1(x62))
        out.append(self.conv47_1(x72))
        out.append(self.conv48_1(x82))
        out.append(self.conv49_1(x92))

        output = torch.zeros(out[0].shape).cuda()
        #accumulate all partial sums
        for out_tensor in out:
            output = output + out_tensor

        x = self.maxpool4(output)
        x = self.batchNorm4(x)
        x = self.hardTanh4(x)

        #x = self.conv5(x)
        #split x
        xp = x
        xp = self.padding(xp)
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x12, x22, x32, x42, x52, x62, x72, x82, x92 = split_tesnsor_256(xp)

        out = []
        out.append(self.conv51(x1))
        out.append(self.conv52(x2))
        out.append(self.conv53(x3))
        out.append(self.conv54(x4))
        out.append(self.conv55(x5))
        out.append(self.conv56(x6))
        out.append(self.conv57(x7))
        out.append(self.conv58(x8))
        out.append(self.conv59(x9))
        out.append(self.conv51_1(x12))
        out.append(self.conv52_1(x22))
        out.append(self.conv53_1(x32))
        out.append(self.conv54_1(x42))
        out.append(self.conv55_1(x52))
        out.append(self.conv56_1(x62))
        out.append(self.conv57_1(x72))
        out.append(self.conv58_1(x82))
        out.append(self.conv59_1(x92))

        output = torch.zeros(out[0].shape).cuda()
        #accumulate all partial sums
        for out_tensor in out:
            output = output + out_tensor

        x = self.batchNorm5(output)
        x = self.hardTanh5(x)

        #x = self.conv6(x)
        #split x
        xp = x
        xp = self.padding(xp)
        x1,x2,x3,x4,x5,x6,x7,x8,x9,x12,x22,x32,x42,x52,x62,x72,x82,x92,x13,x23,x33,x43,x53,x63,x73,x83,x93,x14,x24,x34,x44,x54,x64,x74,x84,x94 = split_tesnsor_512(xp)

        out = []
        out.append(self.conv61(x1))
        out.append(self.conv62(x2))
        out.append(self.conv63(x3))
        out.append(self.conv64(x4))
        out.append(self.conv65(x5))
        out.append(self.conv66(x6))
        out.append(self.conv67(x7))
        out.append(self.conv68(x8))
        out.append(self.conv69(x9))
        out.append(self.conv61_1(x12))
        out.append(self.conv62_1(x22))
        out.append(self.conv63_1(x32))
        out.append(self.conv64_1(x42))
        out.append(self.conv65_1(x52))
        out.append(self.conv66_1(x62))
        out.append(self.conv67_1(x72))
        out.append(self.conv68_1(x82))
        out.append(self.conv69_1(x92))
        out.append(self.conv61_2(x13))
        out.append(self.conv62_2(x23))
        out.append(self.conv63_2(x33))
        out.append(self.conv64_2(x43))
        out.append(self.conv65_2(x53))
        out.append(self.conv66_2(x63))
        out.append(self.conv67_2(x73))
        out.append(self.conv68_2(x83))
        out.append(self.conv69_2(x93))
        out.append(self.conv61_3(x14))
        out.append(self.conv62_3(x24))
        out.append(self.conv63_3(x34))
        out.append(self.conv64_3(x44))
        out.append(self.conv65_3(x54))
        out.append(self.conv66_3(x64))
        out.append(self.conv67_3(x74))
        out.append(self.conv68_3(x84))
        out.append(self.conv69_3(x94))

        output = torch.zeros(out[0].shape).cuda()
        #accumulate all partial sums
        for out_tensor in out:
            output = output + out_tensor

        x = self.maxpool6(output)
        x = self.batchNorm6(x)
        x = self.hardTanh6(x)

        x = x.view(-1, 512 * 4 * 4)

        #x = self.classifier(x)
        x = self.linear7(x)
        x = self.batchNorm7(x)
        x = self.hardTanh7(x)

        x = self.linear8(x)
        x = self.batchNorm8(x)
        x = self.hardTanh8(x)

        x = self.linear9(x)
        x = self.batchNorm9(x)

        return x


def sp_cifar10_model(**kwargs):
    num_classes = kwargs.get( 'num_classes', 10)
    return VGG_Cifar10(num_classes)
