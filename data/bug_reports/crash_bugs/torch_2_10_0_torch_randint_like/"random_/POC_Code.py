import torch
import numpy as np
import torch

def call_func(inputs, low=0, high=None, generator=None, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format):
    return torch.randint_like(inputs, low=low, high=high, generator=generator, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad, memory_format=memory_format)

# Create test input
inputs = torch.randn(3, 4, 5)

# Test parameters that cause the defect
low = 10
high = 10
dtype = torch.int64

print("Testing torch.randint_like with low=10, high=10")
print(f"Input shape: {inputs.shape}")

# Test 1: Dynamic output shape
print("\n1. Dynamic call:")
try:
    dynamic_result = call_func(inputs, low=low, high=high, dtype=dtype)
    print(f"Dynamic output shape: {dynamic_result.shape}")
except Exception as e:
    print(f"Dynamic output shape: {e}")

# Test 2: Static output shape with torch.compile
print("\n2. Static call (torch.compile):")
try:
    compiled_func = torch.compile(call_func, dynamic=True)
    static_result = compiled_func(inputs, low=low, high=high, dtype=dtype)
    print(f"Static output shape: {static_result.shape}")
except Exception as e:
    print(f"Static output shape: {e}")

# Test 3: Meta output shape
print("\n3. Meta call:")
try:
    meta_inputs = inputs.to(device='meta')
    meta_result = call_func(meta_inputs, low=low, high=high, dtype=dtype)
    print(f"Meta output shape: {list(meta_result.shape)}")
except Exception as e:
    print(f"Meta output shape: {e}")