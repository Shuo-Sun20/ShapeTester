import torch
import numpy as np
import torch

def call_func(inputs, alpha=1.):
    return torch.celu_(inputs, alpha)

# Test input that causes the defect
inputs = torch.randn(3, 4)
alpha = 0.0

print("Testing torch.celu_ with alpha=0.0")

# Dynamic output shape (direct call)
try:
    dynamic_result = call_func(inputs.clone(), alpha)
    print(f"Dynamic output shape: {dynamic_result.shape}")
except Exception as e:
    print(f"Dynamic output shape: {type(e).__name__}: {e}")

# Static output shape (compiled)
try:
    compiled_func = torch.compile(call_func, dynamic=True)
    static_result = compiled_func(inputs.clone(), alpha)
    print(f"Static output shape: {static_result.shape}")
except Exception as e:
    print(f"Static output shape: {type(e).__name__}: {e}")

# Meta output shape
try:
    meta_inputs = torch.randn(3, 4, device='meta')
    meta_result = call_func(meta_inputs, alpha)
    print(f"Meta output shape: {meta_result.shape}")
except Exception as e:
    print(f"Meta output shape: {type(e).__name__}: {e}")