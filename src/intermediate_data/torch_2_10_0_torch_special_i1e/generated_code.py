import torch

def call_func(inputs, out=None):
    return torch.special.i1e(inputs, out=out)

input_tensor = torch.randn(3, 4)
example_output = call_func(input_tensor)