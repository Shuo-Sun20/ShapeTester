import torch
import numpy as np
import torch

def call_func(inputs, bias=None, stride=1, padding=0, dilation=1, groups=1):
    input_tensor, weight_tensor = inputs[0], inputs[1]
    output = torch.conv2d(
        input=input_tensor,
        weight=weight_tensor,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups
    )
    return output

# Test input that causes the defect
input_tensor = torch.randn(2, 4, 5, 5)
weight_tensor = torch.randn(6, 4, 3, 3)
inputs = [input_tensor, weight_tensor]

bias = None
stride = 3
padding = 0
dilation = [2, 3]
groups = 1

print("Testing torch.conv2d defect reproduction:")

# Test 1: Direct call (dynamic)
print("\n1. Dynamic output shape (direct call):")
try:
    dynamic_output = call_func(inputs, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    print(f"Dynamic shape: {dynamic_output.shape}")
except Exception as e:
    print(f"Dynamic error: {e}")

# Test 2: Compiled call (static)
print("\n2. Static output shape (torch.compile):")
try:
    compiled_func = torch.compile(call_func, dynamic=True)
    static_output = compiled_func(inputs, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    print(f"Static shape: {static_output.shape}")
except Exception as e:
    print(f"Static error: {e}")

# Test 3: Meta device call
print("\n3. Meta output shape (meta device):")
try:
    meta_input_tensor = torch.randn(2, 4, 5, 5, device='meta')
    meta_weight_tensor = torch.randn(6, 4, 3, 3, device='meta')
    meta_inputs = [meta_input_tensor, meta_weight_tensor]
    meta_output = call_func(meta_inputs, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    print(f"Meta shape: {list(meta_output.shape)}")
except Exception as e:
    print(f"Meta error: {e}")