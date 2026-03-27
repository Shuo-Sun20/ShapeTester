import torch

def call_func(inputs, n, out=None):
    return torch.special.hermite_polynomial_he(inputs, n, out=out)

input_tensor = torch.randn(3, 4)
n_tensor = torch.tensor(2)
example_output = call_func(input_tensor, n_tensor)