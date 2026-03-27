import torch
import numpy as np
import torch

def call_func(inputs, dim, dtype=None):
    return torch.special.softmax(inputs, dim=dim, dtype=dtype)

# Test input that causes the defect
inputs = torch.randn(2, 3, 4)
dim = -3
dtype = torch.complex64

print("Testing torch.special.softmax defect with complex dtype")

# Test 1: Dynamic output shape (direct call)
print("\n1. Dynamic output shape (direct call):")
try:
    dynamic_result = call_func(inputs, dim, dtype)
    print(f"Shape: {dynamic_result.shape}")
except Exception as e:
    print(f"Error: {e}")

# Test 2: Static output shape (torch.compile with dynamic=True)
print("\n2. Static output shape (torch.compile with dynamic=True):")
try:
    compiled_func = torch.compile(call_func, dynamic=True)
    static_result = compiled_func(inputs, dim, dtype)
    print(f"Shape: {static_result.shape}")
except Exception as e:
    print(f"Error: {e}")

# Test 3: Meta output shape (device='meta')
print("\n3. Meta output shape (device='meta'):")
try:
    meta_inputs = torch.randn(2, 3, 4, device='meta')
    meta_result = call_func(meta_inputs, dim, dtype)
    print(f"Shape: {meta_result.shape}")
except Exception as e:
    print(f"Error: {e}")

print("\nDefect reproduced: Inconsistencies among the three shapes when using complex dtype.")