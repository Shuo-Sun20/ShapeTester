import torch

def call_func(inputs, shifts, dims=None):
    return torch.roll(inputs, shifts, dims)

x = torch.randn(4, 2)
example_output = call_func(x, 1, 0)