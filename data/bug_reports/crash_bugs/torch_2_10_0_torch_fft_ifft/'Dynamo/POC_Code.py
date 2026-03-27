import torch
import numpy as np
import torch

def call_func(inputs, n=None, dim=-1, norm=None, out=None):
    return torch.fft.ifft(inputs, n, dim, norm, out=out)

# Create test input
inputs = torch.randn(16, dtype=torch.complex64)

# Test dynamic output shape
dynamic_result = call_func(inputs, n=1, dim=0, norm=None, out=None)
dynamic_shape = list(dynamic_result.shape)
print(f"Dynamic output shape: {dynamic_shape}")

# Test static output shape with torch.compile
try:
    compiled_func = torch.compile(call_func, dynamic=True)
    static_result = compiled_func(inputs, n=1, dim=0, norm=None, out=None)
    static_shape = list(static_result.shape)
    print(f"Static output shape: {static_shape}")
except Exception as e:
    print(f"Static output shape: {e}")

# Test meta output shape
meta_inputs = torch.randn(16, dtype=torch.complex64, device='meta')
meta_result = call_func(meta_inputs, n=1, dim=0, norm=None, out=None)
meta_shape = list(meta_result.shape)
print(f"Meta output shape: {meta_shape}")

# Check for inconsistencies
print(f"\nDefect reproduction:")
print(f"Dynamic shape: {dynamic_shape}")
print(f"Meta shape: {meta_shape}")
print(f"Shapes consistent: {dynamic_shape == meta_shape}")