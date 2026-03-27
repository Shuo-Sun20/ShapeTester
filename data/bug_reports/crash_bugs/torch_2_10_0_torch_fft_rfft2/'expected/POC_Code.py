import torch
import numpy as np
import torch

def call_func(inputs, s=None, dim=(-2, -1), norm=None, out=None):
    input_tensor = inputs[0]  # Extract the single input tensor from the list
    return torch.fft.rfft2(input_tensor, s=s, dim=dim, norm=norm, out=out)

# Test input that causes the defect
inputs = [torch.randn(10, 10)]
s = None
dim = [-1, -2]  # Note: this is different from default (-2, -1)
norm = None
out = None

# Get dynamic output shape
dynamic_result = call_func(inputs, s=s, dim=dim, norm=norm, out=out)
dynamic_shape = list(dynamic_result.shape)
print(f"Dynamic output shape: {dynamic_shape}")

# Get static output shape with torch.compile
try:
    compiled_func = torch.compile(call_func, dynamic=True)
    static_result = compiled_func(inputs, s=s, dim=dim, norm=norm, out=out)
    static_shape = list(static_result.shape)
    print(f"Static output shape: {static_shape}")
except Exception as e:
    print(f"Static output shape error: {e}")

# Get meta output shape
meta_inputs = [torch.randn(10, 10, device='meta')]
meta_result = call_func(meta_inputs, s=s, dim=dim, norm=norm, out=out)
meta_shape = list(meta_result.shape)
print(f"Meta output shape: {meta_shape}")

# Check for inconsistencies
print(f"\nShape comparison:")
print(f"Dynamic: {dynamic_shape}")
print(f"Meta: {meta_shape}")
print(f"Shapes match: {dynamic_shape == meta_shape}")