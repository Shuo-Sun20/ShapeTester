import torch

def call_func(inputs, out=None):
    return torch.special.bessel_j0(inputs[0], out=out)

torch.manual_seed(42)
example_input = torch.randn(5, 5)
example_output = call_func([example_input])