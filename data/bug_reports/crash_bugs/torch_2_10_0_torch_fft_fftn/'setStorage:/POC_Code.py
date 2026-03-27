import torch
import numpy as np
import torch

def call_func(inputs, s=None, dim=None, norm=None, out=None):
    input_tensor = inputs[0]
    return torch.fft.fftn(input_tensor, s=s, dim=dim, norm=norm, out=out)

# Test input that causes the defect
input_tensor = torch.randn(10, 10)
inputs = [input_tensor]
s = [8, 12]
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

# Meta output shape
try:
    meta_input = torch.randn(10, 10, device='meta')
    meta_inputs = [meta_input]
    meta_result = call_func(meta_inputs, s=s, dim=dim, norm=norm, out=out)
    print(f"Meta output shape: {list(meta_result.shape)}")
except Exception as e:
    print(f"Meta output shape: {str(e)}")