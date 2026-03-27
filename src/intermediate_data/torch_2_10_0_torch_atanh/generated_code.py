import torch

def call_func(inputs, out=None):
    return torch.atanh(input=inputs, out=out)

example_input = torch.rand(4) * 2 - 1
example_output = call_func(example_input)