import torch
import numpy as np
import torch

def call_func(inputs, p=None, out=None):
    return torch.linalg.cond(inputs, p, out=out)

# Create test input
inputs = torch.randn(2, 3, 3)
p = 'fro'

print("Testing torch.linalg.cond with different execution modes:")
print(f"Input shape: {inputs.shape}")
print(f"Parameter p: {p}")

# 1. Dynamic output shape (direct call)
try:
    dynamic_result = call_func(inputs, p)
    dynamic_shape = list(dynamic_result.shape)
    print(f"Dynamic output shape: {dynamic_shape}")
except Exception as e:
    print(f"Dynamic execution failed: {e}")

# 2. Static output shape (torch.compile with dynamic=True)
try:
    compiled_func = torch.compile(call_func, dynamic=True)
    static_result = compiled_func(inputs, p)
    static_shape = list(static_result.shape)
    print(f"Static output shape: {static_shape}")
except Exception as e:
    print(f"Static execution failed: {e}")

# 3. Meta output shape (using meta device)
try:
    meta_inputs = inputs.to('meta')
    meta_result = call_func(meta_inputs, p)
    meta_shape = list(meta_result.shape)
    print(f"Meta output shape: {meta_shape}")
except Exception as e:
    print(f"Meta execution failed: {e}")

# Compare shapes to demonstrate the inconsistency
print("\nShape comparison:")
try:
    if 'dynamic_shape' in locals() and 'static_shape' in locals() and 'meta_shape' in locals():
        print(f"Dynamic == Static: {dynamic_shape == static_shape}")
        print(f"Dynamic == Meta: {dynamic_shape == meta_shape}")
        print(f"Static == Meta: {static_shape == meta_shape}")
        
        if not (dynamic_shape == static_shape == meta_shape):
            print("DEFECT REPRODUCED: Shape inconsistency detected!")
except:
    print("Could not compare all shapes due to execution failures")