import torch

def call_func(inputs, rtol=1e-05, atol=1e-08, equal_nan=False):
    return torch.isclose(inputs[0], inputs[1], rtol=rtol, atol=atol, equal_nan=equal_nan)

inputs = [torch.randn(3), torch.randn(3)]
example_output = call_func(inputs, rtol=1e-05, atol=1e-08, equal_nan=False)