import torch
import numpy as np
import torch

def call_func(inputs, s=None, dim=(-2, -1), norm=None, out=None):
    if not isinstance(inputs, list) or len(inputs) != 1:
        raise ValueError("inputs must be a list containing exactly one tensor")
    input_tensor = inputs[0]
    return torch.fft.irfft2(input_tensor, s=s, dim=dim, norm=norm, out=out)

# Test input that causes the defect
input_tensor = torch.randn(10, 5, dtype=torch.complex64)
inputs = [input_tensor]
s = None
dim = [1, 0]
norm = 'backward'
out = None

# Dynamic output shape
dynamic_result = call_func(inputs, s=s, dim=dim, norm=norm, out=out)
print(f"Dynamic output shape: {list(dynamic_result.shape)}")

# Static output shape with torch.compile
compiled_func = torch.compile(call_func, dynamic=True)
static_result = compiled_func(inputs, s=s, dim=dim, norm=norm, out=out)
print(f"Static output shape: {list(static_result.shape)}")

# Meta output shape
try:
    meta_input_tensor = torch.randn(10, 5, dtype=torch.complex64, device='meta')
    meta_inputs = [meta_input_tensor]
    meta_result = call_func(meta_inputs, s=s, dim=dim, norm=norm, out=out)
    print(f"Meta output shape: {list(meta_result.shape)}")
except Exception as e:
    print(f"Meta output shape: {e}")