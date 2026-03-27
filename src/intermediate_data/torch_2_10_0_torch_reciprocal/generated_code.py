import torch

def call_func(inputs, out=None):
    return torch.reciprocal(inputs[0], out=out)

example_input = torch.randn(4)
example_output = call_func([example_input])