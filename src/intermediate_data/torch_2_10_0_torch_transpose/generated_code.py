import torch

def call_func(inputs, dim0, dim1):
    return torch.transpose(inputs[0], dim0, dim1)

x = torch.randn(2, 3)
example_output = call_func([x], 0, 1)