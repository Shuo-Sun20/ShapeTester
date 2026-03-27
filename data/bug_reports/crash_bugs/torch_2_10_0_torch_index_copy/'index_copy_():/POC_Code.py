import torch
import numpy as np
import torch

def call_func(inputs, dim, out=None):
    input_tensor, index_tensor, source_tensor = inputs
    return torch.index_copy(input_tensor, dim, index_tensor, source_tensor, out=out)

# Create test tensors based on the provided shapes
input_tensor = torch.randn(5, 3)
index_tensor = torch.tensor([0, 1, 2], dtype=torch.long)
source_tensor = torch.randn(3, 3)

inputs = [input_tensor, index_tensor, source_tensor]
dim = -1

print("Input shapes:")
print(f"input_tensor: {input_tensor.shape}")
print(f"index_tensor: {index_tensor.shape}")
print(f"source_tensor: {source_tensor.shape}")
print(f"dim: {dim}")

# Test 1: Dynamic output shape (direct call)
print("\n1. Dynamic output shape:")
try:
    dynamic_result = call_func(inputs, dim)
    print(f"Dynamic shape: {dynamic_result.shape}")
except Exception as e:
    print(f"Dynamic error: {e}")

# Test 2: Static output shape (torch.compile with dynamic=True)
print("\n2. Static output shape (torch.compile):")
try:
    compiled_func = torch.compile(call_func, dynamic=True)
    static_result = compiled_func(inputs, dim)
    print(f"Static shape: {static_result.shape}")
except Exception as e:
    print(f"Static error: {e}")

# Test 3: Meta output shape (using meta device)
print("\n3. Meta output shape:")
try:
    meta_input = torch.randn(5, 3, device='meta')
    meta_index = torch.tensor([0, 1, 2], dtype=torch.long, device='meta')
    meta_source = torch.randn(3, 3, device='meta')
    meta_inputs = [meta_input, meta_index, meta_source]
    
    meta_result = call_func(meta_inputs, dim)
    print(f"Meta shape: {meta_result.shape}")
except Exception as e:
    print(f"Meta error: {e}")