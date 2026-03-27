import torch

def call_func(inputs):
    return torch.resolve_neg(inputs)

x = torch.randn(3, dtype=torch.complex64)
y = x.conj()
z = y.imag
example_output = call_func(z)