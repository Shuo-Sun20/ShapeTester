import torch

def call_func(inputs, out=None):
    return torch.special.scaled_modified_bessel_k1(inputs, out=out)

example_output = call_func(torch.randn(5))