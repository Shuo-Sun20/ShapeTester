import torch

def call_func(inputs, out=None):
    return torch.tanh(inputs, out=out)

example_output = call_func(torch.randn(4))