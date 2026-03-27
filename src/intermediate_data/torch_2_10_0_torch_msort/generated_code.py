import torch

def call_func(inputs, out=None):
    return torch.msort(inputs[0], out=out)

t = torch.randn(3, 4)
example_output = call_func([t])