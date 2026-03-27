import torch

def call_func(inputs, n, out=None):
    return torch.special.legendre_polynomial_p(inputs, n, out=out)

input_tensor = torch.randn(4, 3)
n_tensor = torch.tensor(3)
example_output = call_func(input_tensor, n_tensor)