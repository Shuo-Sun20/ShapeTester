import torch

def call_func(inputs, out=None):
    return torch.special.bessel_y0(inputs, out=out)

tensor = torch.randn(3, 3, dtype=torch.float64)
example_output = call_func(tensor)