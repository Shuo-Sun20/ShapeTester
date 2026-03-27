import torch

def call_func(inputs, out=None):
    return torch.special.i0(inputs[0], out=out)

example_output = call_func([torch.randn(3, 3, dtype=torch.float32)])