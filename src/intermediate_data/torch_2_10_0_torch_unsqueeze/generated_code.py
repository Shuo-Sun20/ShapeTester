import torch

def call_func(inputs, dim):
    return torch.unsqueeze(inputs, dim)

# Generate random input tensor
random_tensor = torch.randn(3, 4)
example_output = call_func(random_tensor, 1)