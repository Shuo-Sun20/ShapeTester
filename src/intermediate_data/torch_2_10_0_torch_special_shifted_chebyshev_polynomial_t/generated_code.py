import torch

def call_func(inputs, out=None):
    input_tensor = inputs[0]
    n_tensor = inputs[1]
    return torch.special.shifted_chebyshev_polynomial_t(input_tensor, n_tensor, out=out)

example_inputs = [torch.randn(3), torch.tensor([0, 1, 2])]
example_output = call_func(example_inputs)