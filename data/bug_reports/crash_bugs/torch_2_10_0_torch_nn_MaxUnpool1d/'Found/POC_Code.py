import torch
import numpy as np
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

# Create test inputs based on the provided information
kernel_size = 1
stride = None  # Will default to kernel_size
padding = 0

# Create input tensor and indices with shape [1, 1, 5]
input_tensor = torch.randn(1, 1, 5)
# Create indices with invalid values (index 9 is out of bounds for size 5)
indices = torch.tensor([[[0, 1, 2, 3, 9]]], dtype=torch.long)

inputs = [input_tensor, indices]

print("Input tensor shape:", input_tensor.shape)
print("Indices tensor shape:", indices.shape)
print("Indices values:", indices)

# Test dynamic output shape
try:
    dynamic_result = call_func(kernel_size, stride, inputs, padding)
    print("Dynamic output shape:", dynamic_result.shape)
except Exception as e:
    print("Dynamic output shape error:", str(e))

# Test static output shape with torch.compile
try:
    compiled_func = torch.compile(call_func, dynamic=True)
    static_result = compiled_func(kernel_size, stride, inputs, padding)
    print("Static output shape:", static_result.shape)
except Exception as e:
    print("Static output shape error:", str(e))

# Test meta output shape
try:
    meta_input_tensor = input_tensor.to(device='meta')
    meta_indices = indices.to(device='meta')
    meta_inputs = [meta_input_tensor, meta_indices]
    meta_result = call_func(kernel_size, stride, meta_inputs, padding)
    print("Meta output shape:", meta_result.shape)
except Exception as e:
    print("Meta output shape error:", str(e))