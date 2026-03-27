import torch

def call_func(inputs, dims):
    return torch.permute(inputs[0], dims)

example_inputs = [torch.randn(2, 3, 5)]
example_dims = (2, 0, 1)
example_output = call_func(example_inputs, example_dims)