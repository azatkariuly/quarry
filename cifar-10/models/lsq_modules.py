import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Function
import math

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

def quantize_psum(v, s, p):
    Qn = -2**(p-1)
    Qp = 2**(p-1) - 1

    gradScaleFactor = 1.0 / math.sqrt(v.numel()*Qp)
    s = grad_scale(s, gradScaleFactor)

    vbar = round_pass((v/s).clamp(Qn, Qp))
    vhat = vbar * s

    return vhat

class PartialSumLSQ(nn.Module):
    def __init__(self, **kwargs_q):
        super(PartialSumLSQ, self).__init__()

        self.nbits = kwargs_q['nbits']
        self.step_size = Parameter(torch.Tensor(1))

        #buffer is not updated for optim.step
        self.register_buffer('init_state', torch.zeros(1))

    def forward(self, x):
        if self.init_state == 0:
            self.step_size.data.copy_(2 * x.abs().mean() / math.sqrt(2 ** self.nbits - 1))
            self.init_state.fill_(1)

        x_q = quantize_psum(x, self.step_size, self.nbits)

        return x_q
