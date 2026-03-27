import torch

def call_func(inputs, s=None, dim=None, norm=None, out=None):
    return torch.fft.irfftn(inputs, s, dim, norm, out=out)

t = torch.rand(10, 9)
T = torch.fft.rfftn(t)
example_output = call_func(T, s=t.size())