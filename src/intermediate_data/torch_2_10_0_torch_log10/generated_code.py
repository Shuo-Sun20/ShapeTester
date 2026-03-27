import torch

def call_func(inputs, out=None):
    return torch.log10(inputs, out=out)

example_input = torch.rand(5)
example_output = call_func(example_input)