import torch
import numpy as np
import torch

def call_func(inputs, dim, index):
    input_tensor, src_tensor = inputs
    return torch.select_scatter(input_tensor, src_tensor, dim, index)

# Test input that causes the defect
input_tensor = torch.randn(2, 3, 4)
src_tensor = torch.randn(2, 4)
inputs = [input_tensor, src_tensor]
dim = 0
index = 0

print("Input tensor shape:", input_tensor.shape)
print("Source tensor shape:", src_tensor.shape)
print("Dim:", dim, "Index:", index)

# Dynamic output shape
try:
    dynamic_result = call_func(inputs, dim, index)
    print("Dynamic output shape:", dynamic_result.shape)
except Exception as e:
    print("Dynamic output shape:", str(e))

# Static output shape with torch.compile
try:
    compiled_func = torch.compile(call_func, dynamic=True)
    static_result = compiled_func(inputs, dim, index)
    print("Static output shape:", static_result.shape)
except Exception as e:
    print("Static output shape:", str(e))

# Meta output shape
try:
    meta_input_tensor = torch.randn(2, 3, 4, device='meta')
    meta_src_tensor = torch.randn(2, 4, device='meta')
    meta_inputs = [meta_input_tensor, meta_src_tensor]
    meta_result = call_func(meta_inputs, dim, index)
    print("Meta output shape:", list(meta_result.shape))
except Exception as e:
    print("Meta output shape:", str(e))