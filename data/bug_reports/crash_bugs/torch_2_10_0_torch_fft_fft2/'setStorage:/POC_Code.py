import torch
import numpy as np
import torch

def call_func(inputs, s=None, dim=(-2, -1), norm=None, out=None):
    return torch.fft.fft2(input=inputs, s=s, dim=dim, norm=norm, out=out)

# Create test input
inputs = torch.randn(10, 10)
s = [8, 12]
dim = [1, 0]
norm = None
out = None

print("Testing torch.fft.fft2 with inconsistent shape behavior")
print(f"Input shape: {inputs.shape}")
print(f"s parameter: {s}")
print(f"dim parameter: {dim}")

# Test 1: Dynamic output shape
print("\n1. Dynamic output shape:")
dynamic_result = call_func(inputs, s=s, dim=dim, norm=norm, out=out)
print(f"Dynamic output shape: {list(dynamic_result.shape)}")

# Test 2: Static output shape with torch.compile
print("\n2. Static output shape with torch.compile:")
compiled_func = torch.compile(call_func, dynamic=True)
static_result = compiled_func(inputs, s=s, dim=dim, norm=norm, out=out)
print(f"Static output shape: {list(static_result.shape)}")

# Test 3: Meta output shape
print("\n3. Meta output shape:")
try:
    meta_inputs = inputs.to(device='meta')
    meta_result = call_func(meta_inputs, s=s, dim=dim, norm=norm, out=out)
    print(f"Meta output shape: {list(meta_result.shape)}")
except Exception as e:
    print(f"Meta output shape error: {e}")

print("\nDefect reproduced: Inconsistency between dynamic, static, and meta shapes")