
import numpy as np
import math
import torch


def ulaw2lin(u):
    scale_1 = 32768.0 / 255.0
    u = u - 128
    s = torch.sign(u)
    u = torch.abs(u)
    return s*scale_1*(torch.exp(u/128.*math.log(256))-1)

def lin2ulaw(x):
    scale = 255.0 / 32768.0
    s = torch.sign(x)
    x = torch.abs(x)
    u = (s*(128*torch.log(1+scale*x)/math.log(256)))
    u = torch.clamp(128 + np.round(u), 0, 255)
    return u.short()

def lin2bulw(x, b):
    vm = 2**b
    vm2 = 2**(b-1)
    s1 = (-1.0 + vm)/32768.0
    s = np.sign(x)
    x = np.abs(x)
    y = s*vm2*np.log(1 + s1*x)/np.log(vm)
    u = np.clip(y+vm2, 0, vm-1)
    return u.astype('uint16')

def bulaw2lin(u, b):
    ws = 0.08
    vm = ws*(2**b)
    vm2 = 2**(b - 1)
    u = u - vm2
    s2 = 32768.0/(-1.0+vm)
    s = np.sign(u)
    u = np.abs(u)
    return s*s2*(np.exp(np.log(vm)*u/vm2) - 1)

if __name__ == '__main__':
    u = 10000
    print(lin2bulw(u, 11))

