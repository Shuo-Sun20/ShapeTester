import torch
import numpy as np
import torch

def call_func(inputs, size, stride, storage_offset=None):
    input_tensor = inputs[0]
    return torch.as_strided(input_tensor, size, stride, storage_offset)

# Test input that causes the defect
inputs = [torch.randn(3, 3)]
size = [3, 3]
stride = [2, 1]
storage_offset = 4

print("Testing torch.as_strided defect:")
print(f"Input tensor shape: {inputs[0].shape}")
print(f"Size: {size}")
print(f"Stride: {stride}")
print(f"Storage offset: {storage_offset}")
print()

# Test 1: Dynamic output shape (direct call)
print("1. Dynamic output shape (direct call):")
try:
    dynamic_result = call_func(inputs, size, stride, storage_offset)
    print(f"Dynamic shape: {dynamic_result.shape}")
except Exception as e:
    print(f"Dynamic error: {e}")

# Test 2: Static output shape (torch.compile with dynamic=True)
print("\n2. Static output shape (torch.compile with dynamic=True):")
try:
    compiled_func = torch.compile(call_func, dynamic=True)
    static_result = compiled_func(inputs, size, stride, storage_offset)
    print(f"Static shape: {static_result.shape}")
except Exception as e:
    print(f"Static error: {e}")

# Test 3: Meta output shape (device='meta')
print("\n3. Meta output shape (device='meta'):")
try:
    meta_inputs = [torch.randn(3, 3, device='meta')]
    meta_result = call_func(meta_inputs, size, stride, storage_offset)
    print(f"Meta shape: {meta_result.shape}")
except Exception as e:
    print(f"Meta error: {e}")

print("\nDefect reproduced: Meta device allows invalid storage access while CPU device correctly raises error.")