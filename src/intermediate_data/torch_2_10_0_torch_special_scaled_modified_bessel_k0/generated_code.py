import torch

def call_func(inputs, out=None):
    return torch.special.scaled_modified_bessel_k0(inputs, out=out)

# Generate random input tensor
inputs = torch.randn(3, 4, dtype=torch.float32)
example_output = call_func(inputs)