import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function
from torch.nn.parameter import Parameter

import numpy as np

nbits = 2

class LSQbi(Function):
    @staticmethod
    def forward(self, value, step_size, nbits):
        self.save_for_backward(value, step_size)
        self.other = nbits

        #set levels
        Qn = 0
        Qp = 2**nbits-1

        #v_bar = (value >= 0).type(value.type()) - (value < 0).type(value.type()
        v_bar = (value / step_size).round().clamp(Qn, Qp)
        v_hat = v_bar * step_size

        return v_hat

    @staticmethod
    def backward(self, grad_output):
        value, step_size = self.saved_tensors
        nbits = self.other

        #set levels
        Qn = 0
        Qp = 2**nbits-1
        grad_scale = 1.0 / math.sqrt(value.numel() * Qp)

        v_q = value / step_size
        lower = (v_q < Qn).float()
        higher = (v_q > Qp).float()
        middle = (1.0 - higher - lower)

        #grad_step_size = lower*Qn + higher*Qp + middle*(-value/step_size + (value/step_size).round())
        grad_step_size = lower*Qn + higher*Qp + middle*(-v_q + v_q.round())

        return grad_output*middle, (grad_output*grad_step_size*grad_scale).sum().unsqueeze(dim=0), None


class LSQbw(Function):
    @staticmethod
    def forward(self, value, step_size, nbits):
        self.save_for_backward(value, step_size)
        self.other = nbits

        #set levels
        Qn = -2**(nbits-1)
        Qp = 2**(nbits-1) - 1

        #v_bar = (value >= 0).type(value.type()) - (value < 0).type(value.type()
        v_bar = (value / step_size).round().clamp(Qn, Qp)
        v_hat = v_bar * step_size

        return v_hat

    @staticmethod
    def backward(self, grad_output):
        value, step_size = self.saved_tensors
        nbits = self.other

        #set levels
        Qn = 0
        Qp = 2**nbits-1
        grad_scale = 1.0 / math.sqrt(value.numel() * Qp)

        v_q = value / step_size
        lower = (v_q < Qn).float()
        higher = (v_q > Qp).float()
        middle = (1.0 - higher - lower)

        #grad_step_size = lower*Qn + higher*Qp + middle*(-value/step_size + (value/step_size).round())
        grad_step_size = lower*Qn + higher*Qp + middle*(-v_q + v_q.round())

        return grad_output*middle, (grad_output*grad_step_size*grad_scale).sum().unsqueeze(dim=0), None

class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

        self.alpha = Parameter(torch.ones(1))
        self.beta = Parameter(torch.ones(1))
        self.register_buffer('init_state', torch.zeros(1))

    def forward(self, input):
        if self.init_state == 0:
            init1 = self.weight.abs().mean()
            init2 =  input.abs().mean()

            self.alpha.data.copy_(torch.ones(1).cuda() * init1)
            self.beta.data.copy_(torch.ones(1).cuda() * init2)

            self.init_state.fill_(1)

        if input.size(1) != 784:
            i_q = LSQbi.apply(input, self.beta, nbits)
        else:
            i_q = input

        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()

        w_q = LSQbw.apply(self.weight, self.alpha,nbits)

        out = nn.functional.linear(i_q, w_q)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)

        self.alpha = Parameter(torch.ones(1))
        self.beta = Parameter(torch.ones(1))

        self.register_buffer('init_state', torch.zeros(1))


    def forward(self, input):
        if self.init_state == 0:
            init1 = self.weight.abs().mean()
            init2 =  input.abs().mean()

            self.alpha.data.copy_(torch.ones(1).cuda() * init1)
            self.beta.data.copy_(torch.ones(1).cuda() * init2)

            self.init_state.fill_(1)

        if input.size(1) != 3:
            i_q = LSQbi.apply(input, self.beta, nbits)
        else:
            i_q = input

        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()

        w_q = LSQbw.apply(self.weight, self.alpha, nbits)

        out = nn.functional.conv2d(i_q, w_q, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

