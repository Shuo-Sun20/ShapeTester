import torch

def call_func(inputs):
    return torch.isfinite(inputs)

# Create a tensor with both finite and non-finite values
tensor_data = torch.tensor([1.5, float('inf'), -3.2, float('-inf'), float('nan'), 0.0])
example_output = call_func(tensor_data)