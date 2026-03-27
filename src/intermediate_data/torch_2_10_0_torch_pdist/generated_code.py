import torch

def call_func(inputs, p=2):
    return torch.pdist(inputs, p=p)

# Generate random input tensor
input_tensor = torch.randn(5, 3)  # Shape: N=5, M=3
example_output = call_func(input_tensor, p=2)