import numpy as np
import math
import torch


def _ulaw2lin(u):
    scale_1 = 32768.0 / 255.0
    u = u - 128
    s = torch.sign(u)
    u = torch.abs(u)
    return s*scale_1*(torch.exp(u/128.*math.log(256))-1)

def _lin2ulaw(x):
    scale = 255.0 / 32768.0
    s = torch.sign(x)
    x = torch.abs(x)
    u = (s*(128*torch.log(1+scale*x)/math.log(256)))
    u = torch.clamp(128 + np.round(u), 0, 255)
    return u.short()

def lin2ulw(x, b):
    vm = 2**b
    vm2 = 2**(b-1)
    s1 = (-1.0 + vm)/32768.0
    s = torch.sign(x)
    x = torch.abs(x)
    y = s*vm2*torch.log(1 + s1*x)/torch.log(vm)
    u = torch.clamp(y+vm2, 0, vm-1)
    return u.long()

def ulaw2lin(u, b):
    vm = 2**b
    vm2 = 2**(b - 1)
    u = u - vm2
    s2 = 32768.0/(-1.0+vm)
    s = torch.sign(u)
    u = torch.abs(u)
    return s*s2*(torch.exp(np.log(vm)*u/vm2) - 1)
