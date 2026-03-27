import torch

def call_func(inputs, out=None):
    return torch.special.erf(inputs, out=out)

example_output = call_func(torch.randn(3, 4))