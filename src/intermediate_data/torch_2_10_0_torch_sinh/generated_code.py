import torch

def call_func(inputs, out=None):
    return torch.sinh(inputs, out=out)

example_input = torch.randn(4)
example_output = call_func(example_input)