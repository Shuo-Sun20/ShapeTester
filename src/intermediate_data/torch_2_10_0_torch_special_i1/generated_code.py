import torch

def call_func(inputs, out=None):
    return torch.special.i1(inputs, out=out)

example_output = call_func(torch.randn(5, dtype=torch.float32))