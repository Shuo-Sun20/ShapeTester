import torch
import numpy as np
import torch

def call_func(inputs, out=None):
    input_tensor, other_tensor = inputs
    return torch.matmul(input_tensor, other_tensor, out=out)

# Create test tensors with different dtypes to trigger the defect
input_tensor = torch.randn(10, 3, 4, dtype=torch.float64)  # double precision
other_tensor = torch.randn(4, 5, dtype=torch.float64)      # double precision
out_tensor = torch.empty(10, 3, 5, dtype=torch.float32)   # float precision (mismatch)

inputs = [input_tensor, other_tensor]

print("Input tensor dtype:", input_tensor.dtype)
print("Other tensor dtype:", other_tensor.dtype)
print("Out tensor dtype:", out_tensor.dtype)

# Test dynamic execution
try:
    dynamic_result = call_func(inputs, out=out_tensor)
    print("Dynamic output shape:", dynamic_result.shape)
except Exception as e:
    print("Dynamic output shape:", str(e))

# Test static compilation
try:
    compiled_func = torch.compile(call_func, dynamic=True)
    static_result = compiled_func(inputs, out=out_tensor.clone())
    print("Static output shape:", list(static_result.shape))
except Exception as e:
    print("Static output shape:", str(e))

# Test meta execution
try:
    meta_input = input_tensor.to('meta')
    meta_other = other_tensor.to('meta')
    meta_out = out_tensor.to('meta')
    meta_inputs = [meta_input, meta_other]
    meta_result = call_func(meta_inputs, out=meta_out)
    print("Meta output shape:", list(meta_result.shape))
except Exception as e:
    print("Meta output shape:", str(e))