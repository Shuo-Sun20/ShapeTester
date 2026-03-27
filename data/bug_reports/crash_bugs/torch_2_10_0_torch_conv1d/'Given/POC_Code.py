import torch
import numpy as np
import torch
import torch.nn.functional as F

def call_func(inputs, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return F.conv1d(*inputs, weight, bias, stride, padding, dilation, groups)

# Test input that causes the defect
inputs = [torch.randn(2, 12, 30)]
weight = torch.randn(3, 2, 3)
bias = None
stride = 1
padding = 4
dilation = 5
groups = 6

print("Testing torch.conv1d defect reproduction")
print(f"Input shape: {inputs[0].shape}")
print(f"Weight shape: {weight.shape}")
print(f"Parameters: bias={bias}, stride={stride}, padding={padding}, dilation={dilation}, groups={groups}")

# Test 1: Dynamic output shape (direct call)
print("\n1. Dynamic output shape (direct call):")
try:
    dynamic_result = call_func(inputs, weight, bias, stride, padding, dilation, groups)
    print(f"Dynamic shape: {dynamic_result.shape}")
except Exception as e:
    print(f"Dynamic error: {e}")

# Test 2: Static output shape (torch.compile with dynamic=True)
print("\n2. Static output shape (torch.compile with dynamic=True):")
try:
    compiled_func = torch.compile(call_func, dynamic=True)
    static_result = compiled_func(inputs, weight, bias, stride, padding, dilation, groups)
    print(f"Static shape: {static_result.shape}")
except Exception as e:
    print(f"Static error: {e}")

# Test 3: Meta output shape (device='meta')
print("\n3. Meta output shape (device='meta'):")
try:
    meta_inputs = [torch.randn(2, 12, 30, device='meta')]
    meta_weight = torch.randn(3, 2, 3, device='meta')
    meta_result = call_func(meta_inputs, meta_weight, bias, stride, padding, dilation, groups)
    print(f"Meta shape: {meta_result.shape}")
except Exception as e:
    print(f"Meta error: {e}")

print("\nDefect summary:")
print("- Dynamic and Static calls should fail with groups validation error")
print("- Meta device call produces shape [2, 3, 28] without validation")
print("- This inconsistency indicates a bug in meta device shape inference")