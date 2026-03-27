import torch

def call_func(inputs, diagonal=0, out=None):
    return torch.triu(inputs, diagonal=diagonal, out=out)

example_input = torch.randn(3, 3)
example_output = call_func(example_input, diagonal=0)