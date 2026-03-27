import torch

def call_func(inputs, out=None):
    return torch.sin(inputs, out=out)

a = torch.randn(4)
example_output = call_func(a)