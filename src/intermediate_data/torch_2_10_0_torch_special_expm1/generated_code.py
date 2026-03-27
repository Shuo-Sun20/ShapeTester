import torch

def call_func(inputs, out=None):
    return torch.special.expm1(inputs, out=out)

example_output = call_func(torch.randn(3, 4))