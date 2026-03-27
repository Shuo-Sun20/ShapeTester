import torch

def call_func(inputs, out=None):
    return torch.neg(inputs[0], out=out)

example_input = torch.randn(5)
example_output = call_func([example_input])