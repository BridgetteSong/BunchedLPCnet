def _ulaw2lin(u):
    scale_1 = 32768.0 / 255.0
    u = u - 128
    s = torch.sign(u)
    u = torch.abs(u)
    return s * scale_1 * (torch.exp(u / 128.0 * math.log(256)) - 1)

def _lin2ulaw(x):
    scale = 255.0 / 32768.0
    s = torch.sign(x)
    x = torch.abs(x)
    u = s * 128 * torch.log(1 + scale*x) / math.log(256)
    u = torch.clamp(128 + torch.round(u), 0, 255)
    return u.long()

def lin2ulaw(x, b):
    vm = 2**b
    vm2 = 2**(b-1)
    s1 = (vm - 1.0)/32768.0
    s = torch.sign(x)
    x = torch.abs(x)
    y = s * vm2 * torch.log(1 + s1*x) / math.log(vm)
    u = torch.clamp(torch.round(y) + vm2, 0, vm-1)
    return u.long()

def ulaw2lin(u, b):
    vm = 2**b
    vm2 = 2**(b - 1)
    u = u - vm2
    s2 = 32768.0 / (vm - 1.0)
    s = torch.sign(u)
    u = torch.abs(u)
    return s * s2 * (torch.exp(math.log(vm) * u / vm2) - 1)
