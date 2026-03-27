import torch

def call_func(inputs, dim, index):
    input_tensor = inputs[0]
    output = torch.select(input_tensor, dim, index)
    return output

# Generate random input tensor
input_tensor = torch.randn(3, 4, 5)
dim = 1
index = 2

# Call function and save output
example_output = call_func([input_tensor], dim, index)