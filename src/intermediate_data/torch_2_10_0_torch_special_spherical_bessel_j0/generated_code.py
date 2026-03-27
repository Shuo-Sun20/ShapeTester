import torch

def call_func(inputs, out=None):
    return torch.special.spherical_bessel_j0(inputs, out=out)

example_input = torch.randn(3, 4)
example_output = call_func(example_input)