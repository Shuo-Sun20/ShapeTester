import torch

def call_func(inputs, out=None):
    return torch.special.digamma(inputs, out=out)

input_tensor = torch.rand(4, 3) * 2 + 0.5
example_output = call_func(input_tensor)