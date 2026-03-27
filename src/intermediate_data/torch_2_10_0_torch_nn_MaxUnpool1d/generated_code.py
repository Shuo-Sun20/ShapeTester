import torch
import torch.nn as nn

def call_func(kernel_size, stride, inputs, padding=0):
    unpool_layer = nn.MaxUnpool1d(kernel_size, stride, padding)
    
    if len(inputs) == 2:
        input_tensor, indices = inputs
        return unpool_layer(input_tensor, indices)
    elif len(inputs) == 3:
        input_tensor, indices, output_size = inputs
        return unpool_layer(input_tensor, indices, output_size)
    else:
        raise ValueError("Inputs must contain 2 or 3 elements")

# Construct valid example
pool = nn.MaxPool1d(2, stride=2, return_indices=True)
input_tensor = torch.randn(1, 1, 10)
output, indices = pool(input_tensor)
example_output = call_func(2, 2, [output, indices], 0)