import torch
import numpy as np
import torch

def call_func(inputs, min=None, max=None, out=None):
    return torch.clamp(input=inputs, min=min, max=max, out=out)

# Test input setup
inputs = torch.randn(4)
min_val = -0.5
max_val = torch.tensor(0.5)  # Scalar tensor
out = torch.zeros(4)

# Dynamic output shape
dynamic_result = call_func(inputs, min=min_val, max=max_val, out=out)
dynamic_shape = list(dynamic_result.shape)
print(f"Dynamic output shape: {dynamic_shape}")

# Static output shape with torch.compile
compiled_func = torch.compile(call_func, dynamic=True)
static_result = compiled_func(inputs, min=min_val, max=max_val, out=out)
static_shape = list(static_result.shape)
print(f"Static output shape: {static_shape}")

# Meta output shape
try:
    meta_inputs = torch.randn(4, device='meta')
    meta_max = torch.tensor(0.5, device='meta')
    meta_out = torch.zeros(4, device='meta')
    meta_result = call_func(meta_inputs, min=min_val, max=meta_max, out=meta_out)
    meta_shape = list(meta_result.shape)
    print(f"Meta output shape: {meta_shape}")
except Exception as e:
    print(f"Meta output shape: {e}")

print(f"Shapes consistent: {dynamic_shape == static_shape}")