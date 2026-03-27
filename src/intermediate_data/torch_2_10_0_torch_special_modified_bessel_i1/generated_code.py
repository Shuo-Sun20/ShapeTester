import torch

def call_func(inputs, out=None):
    return torch.special.modified_bessel_i1(inputs, out=out)

example_input = torch.randn(3, 3)
example_output = call_func(example_input)