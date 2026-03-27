import torch

def call_func(inputs, n, out=None):
    input_tensor = inputs
    return torch.special.chebyshev_polynomial_u(input_tensor, n, out=out)

input_tensor = torch.randn(4, 3)
n = 2
example_output = call_func(input_tensor, n)