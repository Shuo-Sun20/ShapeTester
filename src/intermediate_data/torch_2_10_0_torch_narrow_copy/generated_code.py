import torch

def call_func(inputs, dim, start, length, out=None):
    return torch.narrow_copy(inputs[0], dim, start, length, out=out)

x = torch.randn(5, 6, 7)
example_output = call_func([x], dim=1, start=2, length=3)