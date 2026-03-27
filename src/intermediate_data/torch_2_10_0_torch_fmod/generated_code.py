import torch

def call_func(inputs, other, out=None):
    return torch.fmod(inputs, other, out=out)

example_output = call_func(torch.randn(4, 3), torch.randn(4, 3))