import torch
import numpy as np
import torch
import numpy as np
from torch import Tensor

# Define the custom op
@torch.library.custom_op("mylib::numpy_mul", mutates_args=())
def numpy_mul(x: Tensor, *, val: float) -> Tensor:
    x_np = x.cpu().numpy()
    y_np = x_np * val
    return torch.from_numpy(y_np).to(device=x.device)

# Define setup_context function
def setup_context_mul(ctx, inputs, keyword_only_inputs, output):
    ctx.val = keyword_only_inputs["val"]

# Define backward function (note: this has a bug - should return grad * ctx.val)
def backward_sin(ctx, grad):
    return  # This returns None instead of the expected gradient

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

# Test inputs
op_name = 'mylib::numpy_mul'
inputs = [torch.randn(3, requires_grad=True)]
keyword_only_args = {'val': -1.0}

# Get dynamic output shape
dynamic_output = call_func(op_name, backward_sin, inputs, setup_context_mul, keyword_only_args)
dynamic_shape = list(dynamic_output.shape)
print(f"Dynamic output shape: {dynamic_shape}")

# Get static output shape with torch.compile
try:
    compiled_func = torch.compile(lambda: call_func(op_name, backward_sin, inputs, setup_context_mul, keyword_only_args), dynamic=True)
    static_output = compiled_func()
    static_shape = list(static_output.shape)
    print(f"Static output shape: {static_shape}")
except Exception as e:
    print(f"Static output shape: {e}")

# Get meta output shape
meta_inputs = [torch.randn(3, device='meta', requires_grad=True)]
meta_output = call_func(op_name, backward_sin, meta_inputs, setup_context_mul, keyword_only_args)
meta_shape = list(meta_output.shape)
print(f"Meta output shape: {meta_shape}")