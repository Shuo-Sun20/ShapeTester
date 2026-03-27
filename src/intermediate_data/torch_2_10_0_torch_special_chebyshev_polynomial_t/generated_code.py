import torch

def call_func(inputs, out=None):
    if isinstance(inputs, (list, tuple)):
        input_tensor, n = inputs[0], inputs[1]
    else:
        input_tensor, n = inputs, None
    return torch.special.chebyshev_polynomial_t(input_tensor, n, out=out)

example_inputs = (torch.randn(3, 4), torch.tensor(3, dtype=torch.int32))
example_output = call_func(example_inputs)