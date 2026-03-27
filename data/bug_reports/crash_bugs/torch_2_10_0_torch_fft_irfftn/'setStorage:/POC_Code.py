import torch
import numpy as np
import torch

def call_func(inputs, s=None, dim=None, norm=None, out=None):
    return torch.fft.irfftn(inputs, s, dim, norm, out=out)

# Create test input
inputs = torch.randn(10, 5, dtype=torch.complex64)

# Test parameters
s = None
dim = [1, 0]
norm = None
out = None

print("Testing torch.fft.irfftn with inconsistent shapes...")

# Dynamic output shape
try:
    dynamic_result = call_func(inputs, s, dim, norm, out)
    dynamic_shape = list(dynamic_result.shape)
    print(f"Dynamic output shape: {dynamic_shape}")
except Exception as e:
    print(f"Dynamic execution error: {e}")

# Static output shape (compiled)
try:
    compiled_func = torch.compile(call_func, dynamic=True)
    static_result = compiled_func(inputs, s, dim, norm, out)
    static_shape = list(static_result.shape)
    print(f"Static output shape: {static_shape}")
except Exception as e:
    print(f"Static execution error: {e}")

# Meta output shape
try:
    meta_inputs = inputs.to(device='meta')
    meta_result = call_func(meta_inputs, s, dim, norm, out)
    meta_shape = list(meta_result.shape)
    print(f"Meta output shape: {meta_shape}")
except Exception as e:
    print(f"Meta execution error: {e}")