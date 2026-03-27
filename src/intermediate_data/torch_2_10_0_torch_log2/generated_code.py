import torch

def call_func(inputs, out=None):
    return torch.log2(inputs, out=out)

# Generate random input tensor
torch.manual_seed(42)
input_tensor = torch.rand(5)

# Call the function and store output
example_output = call_func(input_tensor)