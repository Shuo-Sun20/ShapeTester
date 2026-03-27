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

# Create a test library
lib = torch.library.Library("test_lib", "FRAGMENT")

# Define custom div operation
@torch.library.custom_op("test_lib::custom_div", mutates_args=())
def custom_div(x: Tensor, y: Tensor) -> Tensor:
    return x / y

# Define vmap function (note: this is misnamed as custom_mul_vmap but used for div)
def custom_mul_vmap(info, in_dims, x, y):
    x_bdim, y_bdim = in_dims
    x = x.movedim(x_bdim, -1) if x_bdim is not None else x.unsqueeze(-1)
    y = y.movedim(y_bdim, -1) if y_bdim is not None else y.unsqueeze(-1)
    result = x / y  # Division operation
    result = result.movedim(-1, 0)
    return result, 0

# Create test inputs
inputs = [torch.randn(3, 4), torch.randn(3, 4)]

# Test dynamic output shape
try:
    dynamic_result = call_func(custom_div, custom_mul_vmap, inputs)
    dynamic_shape = dynamic_result.shape if dynamic_result is not None else "None"
    print(f"Dynamic output shape: {dynamic_shape}")
except Exception as e:
    print(f"Dynamic output shape: Error - {e}")

# Test static output shape with torch.compile
try:
    compiled_func = torch.compile(lambda: call_func(custom_div, custom_mul_vmap, inputs), dynamic=True)
    static_result = compiled_func()
    static_shape = static_result.shape if static_result is not None else "None"
    print(f"Static output shape: {static_shape}")
except Exception as e:
    print(f"Static output shape: Error - {e}")

# Test meta output shape
try:
    meta_inputs = [torch.randn(3, 4, device='meta'), torch.randn(3, 4, device='meta')]
    meta_result = call_func(custom_div, custom_mul_vmap, meta_inputs)
    meta_shape = meta_result.shape if meta_result is not None else "None"
    print(f"Meta output shape: {meta_shape}")
except Exception as e:
    print(f"Meta output shape: {e}")