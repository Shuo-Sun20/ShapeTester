import torch

def call_func(inputs):
    return torch.isinf(inputs)

# Generate random tensor with some infinite values
input_tensor = torch.randn(5)
input_tensor[1] = float('inf')
input_tensor[3] = float('-inf')

example_output = call_func(input_tensor)