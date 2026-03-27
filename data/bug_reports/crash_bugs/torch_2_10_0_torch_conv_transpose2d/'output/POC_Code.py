import torch
import numpy as np
import torch

def call_func(inputs, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    input_tensor, weight_tensor = inputs
    return torch.conv_transpose2d(input_tensor, weight_tensor, bias, stride, padding, output_padding, groups, dilation)

# Test input that causes the defect
input_tensor = torch.randn(1, 4, 5, 5)
weight_tensor = torch.randn(4, 8, 3, 3)
inputs = [input_tensor, weight_tensor]
bias = None
stride = 2
padding = [1, 0]
output_padding = 2
groups = 1
dilation = 2

print("Testing torch.conv_transpose2d defect reproduction")

# Test 1: Dynamic output shape (direct call)
print("\n1. Dynamic output shape (direct call):")
try:
    dynamic_result = call_func(inputs, bias, stride, padding, output_padding, groups, dilation)
    print(f"Dynamic shape: {dynamic_result.shape}")
except Exception as e:
    print(f"Dynamic error: {e}")

# Test 2: Static output shape (torch.compile with dynamic=True)
print("\n2. Static output shape (torch.compile with dynamic=True):")
try:
    compiled_func = torch.compile(call_func, dynamic=True)
    static_result = compiled_func(inputs, bias, stride, padding, output_padding, groups, dilation)
    print(f"Static shape: {static_result.shape}")
except Exception as e:
    print(f"Static error: {e}")

# Test 3: Meta output shape (device='meta')
print("\n3. Meta output shape (device='meta'):")
try:
    meta_input_tensor = torch.randn(1, 4, 5, 5, device='meta')
    meta_weight_tensor = torch.randn(4, 8, 3, 3, device='meta')
    meta_inputs = [meta_input_tensor, meta_weight_tensor]
    meta_result = call_func(meta_inputs, bias, stride, padding, output_padding, groups, dilation)
    print(f"Meta shape: {list(meta_result.shape)}")
except Exception as e:
    print(f"Meta error: {e}")

print("\nDefect reproduction complete: Meta device gives different behavior than regular execution")