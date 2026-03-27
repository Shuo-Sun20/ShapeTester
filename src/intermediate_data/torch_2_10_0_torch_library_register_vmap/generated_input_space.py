import torch
import torch.library
from dataclasses import dataclass, field
from typing import List, Callable, Optional

# Define custom op for testing
lib = torch.library.Library("test_lib", "FRAGMENT")

@torch.library.custom_op("test_lib::custom_mul", mutates_args=())
def custom_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x * y

# Register fake implementation required for torch.compile/export/fx tracing
@custom_mul.register_fake
def custom_mul_fake(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x * y

def call_func(op, vmap_func, inputs):
    # Register vmap implementation
    torch.library.register_vmap(op, vmap_func)
    
    # Call the original op with input tensors
    return op(*inputs)

# Define vmap function for the custom op
def custom_mul_vmap(info, in_dims, x, y):
    x_bdim, y_bdim = in_dims
    x = x.movedim(x_bdim, -1) if x_bdim is not None else x.unsqueeze(-1)
    y = y.movedim(y_bdim, -1) if y_bdim is not None else y.unsqueeze(-1)
    result = x * y
    result = result.movedim(-1, 0)
    return result, 0

# Generate random tensors
torch.manual_seed(42)
tensor1 = torch.randn(3, 4)
tensor2 = torch.randn(3, 4)

# 1. Define valid_test_case
valid_test_case = {
    "op": custom_mul,
    "vmap_func": custom_mul_vmap,
    "inputs": [tensor1, tensor2]
}

# 2. Parameters that can affect output shape: "op" and "vmap_func"

# Define additional operations for value space exploration
@torch.library.custom_op("test_lib::custom_add", mutates_args=())
def custom_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y

@custom_add.register_fake
def custom_add_fake(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y

@torch.library.custom_op("test_lib::custom_sub", mutates_args=())
def custom_sub(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x - y

@custom_sub.register_fake
def custom_sub_fake(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x - y

@torch.library.custom_op("test_lib::custom_matmul", mutates_args=())
def custom_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x @ y

@custom_matmul.register_fake
def custom_matmul_fake(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x @ y

@torch.library.custom_op("test_lib::custom_div", mutates_args=())
def custom_div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x / y

@custom_div.register_fake
def custom_div_fake(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x / y

# Define vmap functions for each operation
def custom_add_vmap(info, in_dims, x, y):
    x_bdim, y_bdim = in_dims
    x = x.movedim(x_bdim, -1) if x_bdim is not None else x.unsqueeze(-1)
    y = y.movedim(y_bdim, -1) if y_bdim is not None else y.unsqueeze(-1)
    result = x + y
    result = result.movedim(-1, 0)
    return result, 0

def custom_sub_vmap(info, in_dims, x, y):
    x_bdim, y_bdim = in_dims
    x = x.movedim(x_bdim, -1) if x_bdim is not None else x.unsqueeze(-1)
    y = y.movedim(y_bdim, -1) if y_bdim is not None else y.unsqueeze(-1)
    result = x - y
    result = result.movedim(-1, 0)
    return result, 0

def custom_matmul_vmap(info, in_dims, x, y):
    x_bdim, y_bdim = in_dims
    x = x.movedim(x_bdim, -1) if x_bdim is not None else x.unsqueeze(-1)
    y = y.movedim(y_bdim, -1) if y_bdim is not None else y.unsqueeze(-1)
    result = x @ y
    result = result.movedim(-1, 0)
    return result, 0

def custom_div_vmap(info, in_dims, x, y):
    x_bdim, y_bdim = in_dims
    x = x.movedim(x_bdim, -1) if x_bdim is not None else x.unsqueeze(-1)
    y = y.movedim(y_bdim, -1) if y_bdim is not None else y.unsqueeze(-1)
    result = x / y
    result = result.movedim(-1, 0)
    return result, 0

# 3. & 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # Discrete parameter: operation function
    op: List[Callable] = field(default_factory=lambda: [
        custom_mul,
        custom_add, 
        custom_sub,
        custom_matmul,
        custom_div
    ])
    
    # Discrete parameter: vmap function (must match operation)
    vmap_func: List[Callable] = field(default_factory=lambda: [
        custom_mul_vmap,
        custom_add_vmap,
        custom_sub_vmap, 
        custom_matmul_vmap,
        custom_div_vmap
    ])