import torch

def call_func(inputs, out=None):
    return torch.sign(inputs, out=out)

example_input = torch.randn(3, 4) * 2 - 1
example_output = call_func(example_input)