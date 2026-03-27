import torch
import numpy as np
import torch

def call_func(inputs, size=None, dtype=None, device=None, pin_memory=False, requires_grad=False, check_invariants=None, is_coalesced=None):
    indices, values = inputs[0], inputs[1]
    return torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=size,
        dtype=dtype,
        device=device,
        pin_memory=pin_memory,
        requires_grad=requires_grad,
        check_invariants=check_invariants,
        is_coalesced=is_coalesced
    )

# Create test inputs
indices = torch.tensor([[0, 1, 1], [2, 0, 2]], dtype=torch.long)
values = torch.tensor([3.0, 4.0, 5.0], dtype=torch.float32)
inputs = [indices, values]

# Test parameters
size = [2, 4]
dtype = torch.float64
device = 'cpu'
pin_memory = False
requires_grad = True
check_invariants = None
is_coalesced = False

print("Testing torch.sparse_coo_tensor shape consistency...")

# Dynamic output shape
try:
    dynamic_result = call_func(inputs, size=size, dtype=dtype, device=device, 
                              pin_memory=pin_memory, requires_grad=requires_grad, 
                              check_invariants=check_invariants, is_coalesced=is_coalesced)
    dynamic_shape = list(dynamic_result.shape)
    print(f"Dynamic output shape: {dynamic_shape}")
except Exception as e:
    print(f"Dynamic execution error: {e}")

# Static output shape (compiled)
try:
    compiled_func = torch.compile(call_func, dynamic=True)
    static_result = compiled_func(inputs, size=size, dtype=dtype, device=device, 
                                 pin_memory=pin_memory, requires_grad=requires_grad, 
                                 check_invariants=check_invariants, is_coalesced=is_coalesced)
    static_shape = list(static_result.shape)
    print(f"Static output shape: {static_shape}")
except Exception as e:
    print(f"Static execution error: {e}")

# Meta output shape
try:
    meta_indices = indices.to(device='meta')
    meta_values = values.to(device='meta')
    meta_inputs = [meta_indices, meta_values]
    meta_result = call_func(meta_inputs, size=size, dtype=dtype, device='meta', 
                           pin_memory=pin_memory, requires_grad=requires_grad, 
                           check_invariants=check_invariants, is_coalesced=is_coalesced)
    meta_shape = list(meta_result.shape)
    print(f"Meta output shape: {meta_shape}")
except Exception as e:
    print(f"Meta output shape: {e}")

# Check for inconsistencies
print("\nShape consistency check:")
try:
    if 'dynamic_shape' in locals() and 'static_shape' in locals():
        if dynamic_shape == static_shape:
            print("Dynamic and static shapes are consistent")
        else:
            print(f"DEFECT: Dynamic shape {dynamic_shape} != Static shape {static_shape}")
    
    if 'meta_shape' in locals():
        if 'dynamic_shape' in locals() and dynamic_shape == meta_shape:
            print("Dynamic and meta shapes are consistent")
        else:
            print(f"DEFECT: Meta execution failed or shapes inconsistent")
    else:
        print("DEFECT: Meta execution failed - Cannot copy out of meta tensor; no data!")
        
except Exception as e:
    print(f"Comparison error: {e}")