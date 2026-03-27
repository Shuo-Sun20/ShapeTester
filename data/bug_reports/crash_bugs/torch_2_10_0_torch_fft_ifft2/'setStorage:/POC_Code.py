import torch
import numpy as np
import torch

def call_func(inputs, s=None, dim=(-2, -1), norm=None, out=None):
    # Extract input tensor from inputs parameter (supporting single tensor or list of tensors)
    if isinstance(inputs, list):
        # Unpack the list to get the actual input tensor
        input_tensor = inputs[0]
    else:
        input_tensor = inputs
    
    # Direct API call for torch.fft.ifft2 function
    result = torch.fft.ifft2(input_tensor, s=s, dim=dim, norm=norm, out=out)
    return result

# Create test input
input_tensor = torch.randn(10, 10, dtype=torch.complex64)

# Test parameters
s = [8, 12]
dim = [1, 0]
norm = None
out = None

print("Testing torch.fft.ifft2 shape inconsistency")
print(f"Input shape: {input_tensor.shape}")
print(f"Parameters: s={s}, dim={dim}, norm={norm}, out={out}")

# Dynamic output shape
dynamic_result = call_func(input_tensor, s=s, dim=dim, norm=norm, out=out)
print(f"Dynamic output shape: {list(dynamic_result.shape)}")

# Static output shape with torch.compile
compiled_func = torch.compile(call_func, dynamic=True)
static_result = compiled_func(input_tensor, s=s, dim=dim, norm=norm, out=out)
print(f"Static output shape: {list(static_result.shape)}")

# Meta output shape
try:
    meta_input = input_tensor.to(device='meta')
    meta_result = call_func(meta_input, s=s, dim=dim, norm=norm, out=out)
    print(f"Meta output shape: {list(meta_result.shape)}")
except Exception as e:
    print(f"Meta output shape error: {str(e)}")