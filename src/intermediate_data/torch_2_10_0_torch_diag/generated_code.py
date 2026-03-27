import torch

def call_func(inputs, diagonal=0, out=None):
    return torch.diag(inputs[0], diagonal=diagonal, out=out)

example_input = [torch.randn(3)]
example_output = call_func(example_input)