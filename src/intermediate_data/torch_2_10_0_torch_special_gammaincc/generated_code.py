import torch

def call_func(inputs, out=None):
    input_tensor, other_tensor = inputs
    return torch.special.gammaincc(input_tensor, other_tensor, out=out)

# Generate random positive tensors meeting the function's requirements
torch.manual_seed(42)
input_tensor = torch.rand(3, 4).abs() + 0.1  # Ensure strictly positive
other_tensor = torch.rand(3, 4).abs() + 0.1  # Ensure strictly positive
inputs = [input_tensor, other_tensor]

example_output = call_func(inputs)