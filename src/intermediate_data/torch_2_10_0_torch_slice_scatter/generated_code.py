import torch

def call_func(inputs, src, dim=0, start=None, end=None, step=1):
    return torch.slice_scatter(inputs, src, dim, start, end, step)

example_output = call_func(torch.randn(8, 8), torch.randn(2, 8), start=6)