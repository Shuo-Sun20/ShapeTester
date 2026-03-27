import torch

def call_func(inputs, k=1, dims=(0, 1)):
    return torch.rot90(inputs[0], k, dims)

x = torch.randn(3, 4)
example_output = call_func([x], 1, (0, 1))