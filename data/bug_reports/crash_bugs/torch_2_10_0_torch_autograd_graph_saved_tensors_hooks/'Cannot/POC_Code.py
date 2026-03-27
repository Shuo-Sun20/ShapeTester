import torch
import numpy as np
import torch

def pack_hook_cpu(tensor):
    return tensor.detach().cpu()

def unpack_hook_example(packed_tensor):
    return packed_tensor

def call_func(pack_hook, unpack_hook, inputs):
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        output = inputs[0] * inputs[1]
    return output

# Test inputs
inputs = [torch.randn(3, requires_grad=True), torch.randn(3, requires_grad=True)]

# Dynamic output shape
dynamic_output = call_func(pack_hook_cpu, unpack_hook_example, inputs)
print(f"Dynamic output shape: {list(dynamic_output.shape)}")

# Static output shape (compiled)
compiled_func = torch.compile(call_func, dynamic=True)
static_output = compiled_func(pack_hook_cpu, unpack_hook_example, inputs)
print(f"Static output shape: {list(static_output.shape)}")

# Meta output shape
meta_inputs = [torch.randn(3, requires_grad=True, device='meta'), torch.randn(3, requires_grad=True, device='meta')]
try:
    meta_output = call_func(pack_hook_cpu, unpack_hook_example, meta_inputs)
    print(f"Meta output shape: {list(meta_output.shape)}")
except Exception as e:
    print(f"Meta output shape: {str(e)}")