import torch
import numpy as np
import torch

def call_func(inputs, s=None, dim=(-2, -1), norm=None, out=None):
    input_tensor = inputs[0]
    return torch.fft.ihfft2(input_tensor, s=s, dim=dim, norm=norm, out=out)

# Create test input
input_tensor = torch.rand(10, 10)
inputs = [input_tensor]

# Test parameters
s = None
dim = [1, 0]
norm = None
out = None

print("Testing torch.fft.ihfft2 with shape inconsistency defect")
print(f"Input shape: {input_tensor.shape}")
print(f"dim parameter: {dim}")

# Dynamic output shape
dynamic_result = call_func(inputs, s=s, dim=dim, norm=norm, out=out)
print(f"Dynamic output shape: {list(dynamic_result.shape)}")

# Static output shape with torch.compile
try:
    compiled_func = torch.compile(call_func, dynamic=True)
    static_result = compiled_func(inputs, s=s, dim=dim, norm=norm, out=out)
    print(f"Static output shape: {list(static_result.shape)}")
except Exception as e:
    print(f"Static output shape error: {e}")

# Meta output shape
meta_input = torch.rand(10, 10, device='meta')
meta_inputs = [meta_input]
meta_result = call_func(meta_inputs, s=s, dim=dim, norm=norm, out=out)
print(f"Meta output shape: {list(meta_result.shape)}")

# Verify the defect
print("\nDefect verification:")
print(f"Dynamic shape: {list(dynamic_result.shape)}")
print(f"Meta shape: {list(meta_result.shape)}")
print(f"Shapes match: {list(dynamic_result.shape) == list(meta_result.shape)}")