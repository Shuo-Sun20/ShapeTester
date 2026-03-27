import torch

def call_func(inputs, dim, index):
    input_tensor, src_tensor = inputs
    return torch.select_scatter(input_tensor, src_tensor, dim, index)

# Generate random tensors
torch.manual_seed(42)
input_tensor = torch.randn(2, 3, 4)  # Shape (2,3,4)
src_tensor = torch.randn(2, 4)       # Shape (2,4) for dim=1, index=1
dim = 1
index = 1

example_output = call_func([input_tensor, src_tensor], dim, index)