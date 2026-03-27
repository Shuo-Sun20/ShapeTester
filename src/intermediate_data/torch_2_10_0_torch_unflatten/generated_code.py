import torch

def call_func(inputs, dim, sizes):
    return torch.unflatten(inputs, dim, sizes)

x = torch.randn(3, 4, 1)
example_output = call_func(x, 1, (2, 2))