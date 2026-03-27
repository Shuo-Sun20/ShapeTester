import torch
import numpy as np
import torch
import torch.nn as nn

def call_func(inputs, dim=None):
    softmin = torch.nn.Softmin(dim=dim)
    output = softmin(inputs)
    return output

# Test input that causes the defect
inputs = torch.randn(2, 3)
dim = -5

print("Testing torch.nn.Softmin with dim=-5 on tensor shape [2, 3]")
print(f"Input shape: {inputs.shape}")
print(f"Dimension parameter: {dim}")

# Test 1: Dynamic output shape (direct call)
print("\n1. Dynamic output shape (direct call):")
try:
    dynamic_output = call_func(inputs, dim=dim)
    print(f"Dynamic output shape: {dynamic_output.shape}")
except Exception as e:
    print(f"Dynamic output shape: {type(e).__name__}: {e}")

# Test 2: Static output shape (torch.compile with dynamic=True)
print("\n2. Static output shape (torch.compile with dynamic=True):")
try:
    compiled_func = torch.compile(call_func, dynamic=True)
    static_output = compiled_func(inputs, dim=dim)
    print(f"Static output shape: {static_output.shape}")
except Exception as e:
    print(f"Static output shape: {type(e).__name__}: {e}")

# Test 3: Meta output shape (device='meta')
print("\n3. Meta output shape (device='meta'):")
try:
    meta_inputs = torch.randn(2, 3, device='meta')
    meta_output = call_func(meta_inputs, dim=dim)
    print(f"Meta output shape: {list(meta_output.shape)}")
except Exception as e:
    print(f"Meta output shape: {type(e).__name__}: {e}")

print("\nDefect Summary:")
print("- Dynamic and Static modes correctly raise IndexError for invalid dimension")
print("- Meta mode incorrectly returns shape [2, 3] without validating dimension bounds")
print("- This inconsistency indicates a bug in meta tensor handling for torch.nn.Softmin")