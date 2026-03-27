import torch

def call_func(inputs):
    return torch.logdet(inputs)

# Generate random square matrices (batch of 2, each 3x3)
input_tensor = torch.randn(2, 3, 3)
example_output = call_func(input_tensor)