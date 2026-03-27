import torch

def call_func(inputs, out=None):
    if isinstance(inputs, list):
        return torch.special.bessel_y1(inputs[0], out=out)
    else:
        return torch.special.bessel_y1(inputs, out=out)

example_output = call_func(torch.rand(3, 4))