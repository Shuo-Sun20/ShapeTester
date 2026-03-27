import torch

def call_func(inputs, out=None):
    return torch.special.airy_ai(inputs, out=out)

inputs = torch.randn(3, 4)
example_output = call_func(inputs)