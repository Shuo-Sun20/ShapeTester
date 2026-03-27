import torch
import numpy as np
import torch

def call_func(inputs, s=None, dim=(-2, -1), norm=None, out=None):
    input_tensor = inputs[0]  # Extract the single input tensor from the list
    return torch.fft.rfft2(input_tensor, s=s, dim=dim, norm=norm, out=out)

# Test input that causes the defect
inputs = [torch.randn(10, 10)]
s = [3, 8]
dim = [1, 0]
norm = None
out = None

# Dynamic output shape
dynamic_result = call_func(inputs, s=s, dim=dim, norm=norm, out=out)
print(f"Dynamic output shape: {list(dynamic_result.shape)}")

# Static output shape with torch.compile
compiled_func = torch.compile(call_func, dynamic=True)
static_result = compiled_func(inputs, s=s, dim=dim, norm=norm, out=out)
print(f"Static output shape: {list(static_result.shape)}")

# Meta shape
try:
    meta_inputs = [torch.randn(10, 10, device='meta')]
    meta_result = call_func(meta_inputs, s=s, dim=dim, norm=norm, out=out)
    print(f"Meta output shape: {list(meta_result.shape)}")
except Exception as e:
    print(f"Meta output shape: {e}")