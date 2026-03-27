import torch

def call_func(inputs, n, out=None):
    return torch.linalg.matrix_power(inputs, n, out=out)

example_output = call_func(torch.randn(3, 3), 3)