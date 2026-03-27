import torch
import numpy as np
import torch

def call_func(kernel_size, stride, inputs, padding=0):
    unpool = torch.nn.MaxUnpool2d(kernel_size, stride, padding)
    return unpool(*inputs)

# Test input that causes the defect
kernel_size = [1, 1]
stride = 5
padding = [3, 4]
inputs = [torch.randn(1, 1, 2, 2), torch.randint(0, 1, (1, 1, 2, 2))]

# Dynamic output shape
dynamic_output = call_func(kernel_size, stride, inputs, padding)
print(f"Dynamic output shape: {list(dynamic_output.shape)}")

# Static output shape with torch.compile
compiled_func = torch.compile(call_func, dynamic=True)
static_output = compiled_func(kernel_size, stride, inputs, padding)
print(f"Static output shape: {list(static_output.shape)}")

# Meta output shape
meta_inputs = [torch.randn(1, 1, 2, 2, device='meta'), torch.randint(0, 1, (1, 1, 2, 2), device='meta')]
try:
    meta_output = call_func(kernel_size, stride, meta_inputs, padding)
    print(f"Meta output shape: {list(meta_output.shape)}")
except Exception as e:
    print(f"Meta output shape: {str(e)}")