import torch

def call_func(inputs, dim=None):
    return torch.count_nonzero(inputs[0], dim=dim)

x = torch.randn(4, 5)
example_output = call_func([x], dim=1)