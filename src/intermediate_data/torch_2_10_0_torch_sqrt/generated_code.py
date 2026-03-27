import torch

def call_func(inputs, out=None):
    return torch.sqrt(inputs, out=out)

example_input = torch.rand(4)
example_output = call_func(example_input)