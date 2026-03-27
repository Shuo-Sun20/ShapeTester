import torch
import numpy as np
import torch

def call_func(inputs, dim, index):
    input_tensor, src_tensor = inputs
    return torch.select_scatter(input_tensor, src_tensor, dim, index)

# Test input that causes the defect
input_tensor = torch.randn(4, 3, 2)
src_tensor = torch.randn(3, 2)
inputs = [input_tensor, src_tensor]
dim = 2
index = 2

print("Input tensor shape:", input_tensor.shape)
print("Source tensor shape:", src_tensor.shape)
print("Dim:", dim, "Index:", index)
print()

# Dynamic output shape
print("=== Dynamic output shape ===")
try:
    dynamic_result = call_func(inputs, dim, index)
    print("Dynamic shape:", dynamic_result.shape)
except Exception as e:
    print("Dynamic error:", str(e))

# Static output shape with torch.compile
print("\n=== Static output shape (torch.compile) ===")
try:
    compiled_func = torch.compile(call_func, dynamic=True)
    static_result = compiled_func(inputs, dim, index)
    print("Static shape:", static_result.shape)
except Exception as e:
    print("Static error:", str(e))

# Meta output shape
print("\n=== Meta output shape ===")
try:
    meta_input_tensor = torch.randn(4, 3, 2, device='meta')
    meta_src_tensor = torch.randn(3, 2, device='meta')
    meta_inputs = [meta_input_tensor, meta_src_tensor]
    meta_result = call_func(meta_inputs, dim, index)
    print("Meta shape:", meta_result.shape)
except Exception as e:
    print("Meta error:", str(e))