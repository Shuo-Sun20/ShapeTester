import torch

def call_func(inputs, out=None):
    return torch.kron(inputs[0], inputs[1], out=out)

example_output = call_func([torch.randn(2, 2), torch.randn(2, 2)])