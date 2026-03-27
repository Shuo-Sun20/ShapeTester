import torch
import numpy as np
import torch

def call_func(inputs, size, stride, storage_offset=None):
    input_tensor, src_tensor = inputs
    return torch.as_strided_scatter(
        input=input_tensor,
        src=src_tensor,
        size=size,
        stride=stride,
        storage_offset=storage_offset
    )

# Create test tensors
input_tensor = torch.zeros(3, 3)
src_tensor = torch.ones(3, 3)
inputs = [input_tensor, src_tensor]
size = [3, 3]
stride = [1, 2]
storage_offset = 3

print("Input tensor shape:", input_tensor.shape)
print("Src tensor shape:", src_tensor.shape)
print("Size:", size)
print("Stride:", stride)
print("Storage offset:", storage_offset)

# Test 1: Dynamic output shape (direct call)
print("\n=== Dynamic output shape (direct call) ===")
try:
    dynamic_result = call_func(inputs, size, stride, storage_offset)
    print("Dynamic output shape:", dynamic_result.shape)
except Exception as e:
    print("Dynamic error:", str(e))

# Test 2: Static output shape (torch.compile with dynamic=True)
print("\n=== Static output shape (torch.compile with dynamic=True) ===")
try:
    compiled_func = torch.compile(call_func, dynamic=True)
    static_result = compiled_func(inputs, size, stride, storage_offset)
    print("Static output shape:", static_result.shape)
except Exception as e:
    print("Static error:", str(e))

# Test 3: Meta output shape (device='meta')
print("\n=== Meta output shape (device='meta') ===")
try:
    meta_input_tensor = torch.zeros(3, 3, device='meta')
    meta_src_tensor = torch.ones(3, 3, device='meta')
    meta_inputs = [meta_input_tensor, meta_src_tensor]
    meta_result = call_func(meta_inputs, size, stride, storage_offset)
    print("Meta output shape:", meta_result.shape)
except Exception as e:
    print("Meta error:", str(e))