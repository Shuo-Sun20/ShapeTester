import torch

def call_func(inputs):
    return torch.trace(inputs)

inputs = torch.randn(4, 4)
example_output = call_func(inputs)