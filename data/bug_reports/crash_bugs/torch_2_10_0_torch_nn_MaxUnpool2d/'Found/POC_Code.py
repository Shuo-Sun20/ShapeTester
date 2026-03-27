import torch
import numpy as np
import torch

def call_func(kernel_size, stride, inputs, padding=0):
    unpool = torch.nn.MaxUnpool2d(kernel_size, stride, padding)
    return unpool(*inputs)

# Create test inputs that cause the defect
# The issue appears to be with invalid indices in the MaxUnpool2d operation
input_tensor = torch.randn(1, 1, 2, 2)
# Create indices tensor with invalid values that exceed the valid range
indices_tensor = torch.tensor([[[[0, 14], [2, 3]]]], dtype=torch.long)  # 14 is invalid for 2x2 output

inputs = [input_tensor, indices_tensor]

kernel_size = 1
stride = 1
padding = 0

print("Testing torch.nn.MaxUnpool2d defect:")
print(f"Input tensor shape: {input_tensor.shape}")
print(f"Indices tensor: {indices_tensor}")

# Test 1: Dynamic output shape (direct call)
try:
    dynamic_result = call_func(kernel_size, stride, inputs, padding)
    print(f"Dynamic output shape: {dynamic_result.shape}")
except Exception as e:
    print(f"Dynamic output shape: {e}")

# Test 2: Static output shape (compiled with dynamic=True)
try:
    compiled_func = torch.compile(call_func, dynamic=True)
    static_result = compiled_func(kernel_size, stride, inputs, padding)
    print(f"Static output shape: {static_result.shape}")
except Exception as e:
    print(f"Static output shape: {e}")

# Test 3: Meta shape (using meta device)
try:
    meta_input = input_tensor.to('meta')
    meta_indices = indices_tensor.to('meta')
    meta_inputs = [meta_input, meta_indices]
    meta_result = call_func(kernel_size, stride, meta_inputs, padding)
    print(f"Meta output shape: {list(meta_result.shape)}")
except Exception as e:
    print(f"Meta output shape: {e}")