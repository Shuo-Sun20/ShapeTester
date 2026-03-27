import torch

def call_func(inputs, n, out=None):
    return torch.special.chebyshev_polynomial_w(inputs, n, out=out)

example_input = torch.randn(3, 4)
example_n = torch.tensor(2, dtype=torch.int32)
example_output = call_func(example_input, example_n)