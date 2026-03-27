import torch

def call_func(inputs, out=None):
    return torch.special.bessel_j1(inputs, out=out)

example_output = call_func(torch.randn(2, 3))