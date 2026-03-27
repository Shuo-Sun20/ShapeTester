import torch

def call_func(inputs, out=None):
    input_tensor, n_tensor = inputs
    return torch.special.shifted_chebyshev_polynomial_v(input_tensor, n_tensor, out=out)

torch.manual_seed(0)
input_tensor = torch.randn(3, 4)
n_tensor = torch.tensor([2])
example_output = call_func([input_tensor, n_tensor])