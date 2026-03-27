import torch

def call_func(inputs, out=None):
    return torch.frac(input=inputs, out=out)

example_input = torch.randn(3, 4)
example_output = call_func(example_input)