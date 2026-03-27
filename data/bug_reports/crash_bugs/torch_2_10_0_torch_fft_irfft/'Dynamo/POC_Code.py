import torch
import numpy as np
import torch

def call_func(inputs, n=None, dim=-1, norm=None, out=None):
    return torch.fft.irfft(input=inputs, n=n, dim=dim, norm=norm, out=out)

# Create test input - complex tensor with shape [3]
inputs = torch.randn(3, dtype=torch.complex64)

# Test parameters
n = 1
dim = -1
norm = None
out = None

print("=== Testing torch.fft.irfft shape consistency ===")

# 1. Dynamic output shape
print("\n1. Dynamic execution:")
try:
    dynamic_result = call_func(inputs, n=n, dim=dim, norm=norm, out=out)
    print(f"Dynamic output shape: {list(dynamic_result.shape)}")
except Exception as e:
    print(f"Dynamic execution failed: {e}")

# 2. Static output shape with torch.compile
print("\n2. Static execution with torch.compile:")
try:
    compiled_func = torch.compile(call_func, dynamic=True)
    static_result = compiled_func(inputs, n=n, dim=dim, norm=norm, out=out)
    print(f"Static output shape: {list(static_result.shape)}")
except Exception as e:
    print(f"Static execution failed: {e}")

# 3. Meta output shape
print("\n3. Meta execution:")
try:
    meta_inputs = inputs.to(device='meta')
    meta_result = call_func(meta_inputs, n=n, dim=dim, norm=norm, out=out)
    print(f"Meta output shape: {list(meta_result.shape)}")
except Exception as e:
    print(f"Meta execution failed: {e}")

print("\n=== Shape comparison ===")
try:
    dynamic_shape = list(dynamic_result.shape)
    meta_shape = list(meta_result.shape)
    
    print(f"Dynamic shape: {dynamic_shape}")
    print(f"Meta shape: {meta_shape}")
    
    if dynamic_shape == meta_shape:
        print("✓ Dynamic and Meta shapes are consistent")
    else:
        print("✗ Dynamic and Meta shapes are inconsistent!")
        
except:
    print("Could not compare shapes due to execution failures")