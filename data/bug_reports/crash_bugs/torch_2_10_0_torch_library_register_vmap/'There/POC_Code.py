import torch
import numpy as np
import torch
from torch import Tensor
from typing import Tuple

def call_func(op, vmap_func, inputs):
    # Register vmap implementation
    torch.library.register_vmap(op, vmap_func)
    
    # Call the original op with input tensors
    return op(*inputs)

# Create a custom library
lib = torch.library.Library("test_lib", "FRAGMENT")

# Define a custom operation
@torch.library.custom_op("test_lib::custom_div", mutates_args=())
def custom_div(x: Tensor, y: Tensor) -> Tensor:
    return x / y

# Define vmap function with incorrect name (should be custom_div_vmap but using custom_add_vmap)
def custom_add_vmap(info, in_dims, x, y):
    x_bdim, y_bdim = in_dims
    x = x.movedim(x_bdim, -1) if x_bdim is not None else x.unsqueeze(-1)
    y = y.movedim(y_bdim, -1) if y_bdim is not None else y.unsqueeze(-1)
    result = x / y  # Should be division to match the op
    result = result.movedim(-1, 0)
    return result, 0

# Test inputs
inputs = [torch.randn(3, 4), torch.randn(3, 4)]

# Dynamic output shape
dynamic_result = call_func(custom_div, custom_add_vmap, inputs)
print(f"Dynamic output shape: {list(dynamic_result.shape)}")

# Static output shape with torch.compile
compiled_func = torch.compile(call_func, dynamic=True)
try:
    static_result = compiled_func(custom_div, custom_add_vmap, inputs)
    print(f"Static output shape: {list(static_result.shape)}")
except Exception as e:
    print(f"Static compilation error: {e}")

# Meta output shape
meta_inputs = [torch.randn(3, 4, device='meta'), torch.randn(3, 4, device='meta')]
try:
    meta_result = call_func(custom_div, custom_add_vmap, meta_inputs)
    print(f"Meta output shape: {list(meta_result.shape)}")
except Exception as e:
    print(f"Meta output shape: {e}")