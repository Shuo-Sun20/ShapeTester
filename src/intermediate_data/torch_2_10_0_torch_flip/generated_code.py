import torch

def call_func(inputs, dims):
    return torch.flip(inputs, dims)

example_input = torch.randn(2, 3, 4)
example_output = call_func(example_input, [0, 1])