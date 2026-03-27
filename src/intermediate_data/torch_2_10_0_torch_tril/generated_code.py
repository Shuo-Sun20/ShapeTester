import torch

def call_func(inputs, diagonal=0, out=None):
    return torch.tril(inputs, diagonal=diagonal, out=out)

b = torch.randn(4, 6)
example_output = call_func(b, diagonal=1)