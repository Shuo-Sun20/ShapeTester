import torch

def call_func(inputs, out=None):
    input_tensor, n_tensor = inputs
    return torch.special.shifted_chebyshev_polynomial_w(input_tensor, n_tensor, out=out)

input_tensor = torch.randn(3, 4)
n_tensor = torch.tensor(2)
example_output = call_func([input_tensor, n_tensor])