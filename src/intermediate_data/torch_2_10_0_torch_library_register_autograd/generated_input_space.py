import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable

# Define the custom op and functions for numpy_sin
@torch.library.custom_op("mylib::numpy_sin", mutates_args=())
def numpy_sin(x: torch.Tensor) -> torch.Tensor:
    x_np = x.cpu().numpy()
    y_np = np.sin(x_np)
    return torch.from_numpy(y_np).to(device=x.device)

def setup_context_sin(ctx, inputs, output):
    x, = inputs
    ctx.save_for_backward(x)

def backward_sin(ctx, grad):
    x, = ctx.saved_tensors
    return grad * x.cos()

# Register fake implementation for numpy_sin
def numpy_sin_fake(x):
    return torch.empty_like(x)
numpy_sin.register_fake(numpy_sin_fake)

# Define the custom op and functions for numpy_mul
@torch.library.custom_op("mylib::numpy_mul", mutates_args=())
def numpy_mul(x: torch.Tensor, *, val: float) -> torch.Tensor:
    x_np = x.cpu().numpy()
    y_np = x_np * val
    return torch.from_numpy(y_np).to(device=x.device)

def setup_context_mul(ctx, inputs, keyword_only_inputs, output):
    ctx.val = keyword_only_inputs["val"]

def backward_mul(ctx, grad):
    return grad * ctx.val

# Register fake implementation for numpy_mul
def numpy_mul_fake(x, *, val):
    return torch.empty_like(x)
numpy_mul.register_fake(numpy_mul_fake)

def call_func(op_name, backward_fn, inputs, setup_context_fn=None, keyword_only_args=None):
    if setup_context_fn is not None:
        if keyword_only_args is not None:
            torch.library.register_autograd(op_name, backward_fn, setup_context=setup_context_fn)
        else:
            torch.library.register_autograd(op_name, backward_fn, setup_context=setup_context_fn)
    else:
        torch.library.register_autograd(op_name, backward_fn)
    
    lib_name, op_name_suffix = op_name.split("::")
    op = getattr(getattr(torch.ops, lib_name), op_name_suffix)
    
    if keyword_only_args is not None:
        return op(*inputs, **keyword_only_args)
    else:
        return op(*inputs)

# 1. valid_test_case
valid_test_case = {
    "op_name": "mylib::numpy_sin",
    "backward_fn": backward_sin,
    "inputs": [torch.randn(3, requires_grad=True)],
    "setup_context_fn": setup_context_sin,
    "keyword_only_args": None
}

# 2. Parameters affecting output shape (except "inputs"):
# op_name, backward_fn, setup_context_fn, keyword_only_args

# 3. Value spaces (discretized where applicable)
# op_name: discrete
op_name_values = ["mylib::numpy_sin", "mylib::numpy_mul"]

# backward_fn: discrete (function objects)
backward_fn_values = [backward_sin, backward_mul]

# setup_context_fn: discrete (function objects, None)
setup_context_fn_values = [setup_context_sin, setup_context_mul, None]

# keyword_only_args: discrete (None or dict with float values)
keyword_only_args_values = [
    None,
    {"val": -10.0},
    {"val": -1.0},
    {"val": 0.0},
    {"val": 1.0},
    {"val": 10.0}
]

# 4. InputSpace dataclass
@dataclass
class InputSpace:
    op_name: List[str] = field(default_factory=lambda: op_name_values)
    backward_fn: List[Callable] = field(default_factory=lambda: backward_fn_values)
    setup_context_fn: List[Optional[Callable]] = field(default_factory=lambda: setup_context_fn_values)
    keyword_only_args: List[Optional[Dict[str, float]]] = field(default_factory=lambda: keyword_only_args_values)