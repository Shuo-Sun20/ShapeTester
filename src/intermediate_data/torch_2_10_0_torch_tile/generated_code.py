import torch

def call_func(inputs, dims):
    return torch.tile(inputs, dims)

# Create a random 2x3 tensor as input
input_tensor = torch.randn(2, 3)
example_output = call_func(input_tensor, (2, 3))