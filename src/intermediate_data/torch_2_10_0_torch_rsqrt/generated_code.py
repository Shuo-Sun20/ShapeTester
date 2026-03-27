import torch

def call_func(inputs, out=None):
    return torch.rsqrt(inputs, out=out)

a = torch.rand(4) + 0.1  # Add 0.1 to avoid division by zero/sqrt(0)
example_output = call_func(a)