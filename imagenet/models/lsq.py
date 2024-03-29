import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Function
import math

class LSQ(Function):
    @staticmethod
    def forward(self, value, step_size, nbits, signed):
        self.save_for_backward(value, step_size)
        self.other = nbits, signed

        #set levels
        if signed:
            Qn = -2**(nbits-1)
            Qp = 2**(nbits-1) - 1
        else:
            Qn = 0
            Qp = 2**nbits - 1

        v_bar = (value/step_size).round().clamp(Qn, Qp)
        v_hat = v_bar*step_size
        return v_hat

    @staticmethod
    def backward(self, grad_output):
        value, step_size = self.saved_tensors
        nbits, signed = self.other

        #set levels
        if signed:
            Qn = -2**(nbits-1)
            Qp = 2**(nbits-1) - 1
        else:
            Qn = 0
            Qp = 2**nbits - 1

        grad_scale = 1.0 / math.sqrt(value.numel() * Qp)

        lower = (value/step_size <= Qn).float()
        higher = (value/step_size >= Qp).float()
        middle = (1.0 - higher - lower)

        grad_step_size = lower*Qn + higher*Qp + middle*(-value/step_size + (value/step_size).round())

        return grad_output*middle, (grad_output*grad_step_size*grad_scale).sum().unsqueeze(dim=0), None, None

class LSQ_binary(Function):
    @staticmethod
    def forward(self, value, step_size):
        self.save_for_backward(value, step_size)

        v_bar = (value/step_size).sign() #(value/step_size).ceil().clamp(Qn, Qp)
        v_hat = v_bar*step_size #(2**(step_size.log2().round()))
        return v_hat

    @staticmethod
    def backward(self, grad_output):
        value, step_size = self.saved_tensors

        #set levels
        Qn = -1
        Qp = 1

        grad_scale = 1.0 / math.sqrt(value.numel() * Qp)

        lower = (value/step_size <= Qn).float()
        higher = (value/step_size >= Qp).float()
        middle = (1.0 - higher - lower)

        grad_step_size = lower*Qn + higher*Qp + middle*(-value/step_size + (value/step_size).sign())

        return grad_output*middle, (grad_output*grad_step_size*grad_scale).sum().unsqueeze(dim=0)

def grad_scale(x, scale):
    yOut = x
    yGrad = x*scale
    y = yOut.detach() - yGrad.detach() + yGrad
    return y

def round_pass(x):
    yOut = x.round()
    yGrad = x
    y = yOut.detach() - yGrad.detach() + yGrad
    return y

def dsf_round_shift_pass(s):
    yOut = 2**(s.log2().round())
    yGrad = s
    y = yOut.detach() - yGrad.detach() + yGrad
    return y

def quantizeLSQ(v, s, p, isActivation=False, k=8):
    if isActivation:
        Qn = 0
        Qp = 2**p - 1
        gradScaleFactor = 1.0 / math.sqrt(v.numel() * Qp)
    else: # is weight
        Qn = -2**(p-1)
        Qp = 2**(p-1) - 1
        gradScaleFactor = 1.0 / math.sqrt(v.numel() * Qp)

    #quantize layer
    s = grad_scale(s, gradScaleFactor)
    vbar = round_pass((v/s).clamp(Qn, Qp))

    # vhat = vbar*dsf_round_shift_pass(s)
    vhat = vbar*s
    #print(2**s_dsf)

    return vhat

class Conv2dLSQ(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super(Conv2dLSQ, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                        padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.nbits = kwargs['nbits']
        self.step_size = Parameter(torch.Tensor(1))

        #buffer is not updated for optim.step
        self.register_buffer('init_state', torch.zeros(1))

    def forward(self, x):
        if self.init_state == 0:
            self.step_size.data.copy_(2 * self.weight.abs().mean() / math.sqrt(2 ** (self.nbits - 1) - 1))
            self.init_state.fill_(1)

        #w_q = quantizeLSQ(self.weight, self.step_size, self.nbits)
        w_q = LSQ.apply(self.weight, self.step_size, self.nbits, True)

        return F.conv2d(x, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class LinearLSQ(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super(LinearLSQ, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.nbits = kwargs['nbits']
        self.step_size = Parameter(torch.Tensor(1))

        #buffer is not updated for optim.step
        self.register_buffer('init_state', torch.zeros(1))

    def forward(self, x):
        if self.init_state == 0:
            self.step_size.data.copy_(2 * self.weight.abs().mean() / math.sqrt(2 ** (self.nbits - 1) - 1))
            self.init_state.fill_(1)

        #w_q = quantizeLSQ(self.weight, self.step_size, self.nbits)
        w_q = LSQ.apply(self.weight, self.step_size, self.nbits, True)

        return F.linear(x, w_q, self.bias)

class ActLSQ(nn.Module):
    def __init__(self, **kwargs):
        super(ActLSQ, self).__init__()
        self.nbits = kwargs['nbits']
        self.step_size = Parameter(torch.Tensor(1))

        #buffer is not updated for optim.step
        self.register_buffer('init_state', torch.zeros(1))

    def forward(self, x):
        if self.init_state == 0:
            self.step_size.data.copy_(2 * x.abs().mean() / math.sqrt(2 ** self.nbits - 1))
            self.init_state.fill_(1)

        #x_q = quantizeLSQ(x, self.step_size, self.nbits, isActivation=True)
        x_q = LSQ.apply(x, self.step_size, self.nbits, False)

        return x_q

class PartialSumLSQ(nn.Module):
    def __init__(self, **kwargs):
        super(PartialSumLSQ, self).__init__()
        self.psumq_bits = kwargs['psumq_bits']
        self.dsf_bits = kwargs['dsf_bits']
        self.step_size = Parameter(torch.Tensor(1))

        #buffer is not updated for optim.step
        self.register_buffer('init_state', torch.zeros(1))

    def forward(self, x):
        if self.init_state == 0:
            if self.psumq_bits == 1:
                self.step_size.data.copy_(2 * x.abs().mean())
            else:
                self.step_size.data.copy_(2 * x.abs().mean() / math.sqrt(2 ** self.psumq_bits - 1))
            self.init_state.fill_(1)

        if self.psumq_bits == 1:
            x_q = LSQ_binary.apply(x, self.step_size)
        else:
            x_q = quantizeLSQ(x, self.step_size, self.psumq_bits, k=self.dsf_bits)

        return x_q
