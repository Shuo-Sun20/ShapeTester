import torch
import numpy as np
import torch
import torch.nn.functional as F

def call_func(inputs, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    if len(inputs) == 2:
        input_tensor, weight = inputs
        bias = None
    elif len(inputs) == 3:
        input_tensor, weight, bias = inputs
    else:
        raise ValueError("inputs list must contain 2 (input, weight) or 3 (input, weight, bias) tensors")
    
    return F.conv_transpose3d(
        input_tensor, 
        weight, 
        bias=bias, 
        stride=stride, 
        padding=padding, 
        output_padding=output_padding, 
        groups=groups, 
        dilation=dilation
    )

# Create test inputs
input_tensor = torch.randn(2, 32, 10, 10, 10)
weight = torch.randn(32, 2, 3, 3, 3)
inputs = [input_tensor, weight]

# Test parameters
stride = [1, 2, 3]
padding = 0
output_padding = 2
groups = 16
dilation = 2

print("Testing torch.conv_transpose3d defect:")
print(f"Input shape: {input_tensor.shape}")
print(f"Weight shape: {weight.shape}")
print(f"Parameters: stride={stride}, padding={padding}, output_padding={output_padding}, groups={groups}, dilation={dilation}")

# Test 1: Dynamic output shape (direct call)
print("\n1. Dynamic output shape (direct call):")
try:
    dynamic_output = call_func(inputs, stride=stride, padding=padding, output_padding=output_padding, groups=groups, dilation=dilation)
    print(f"Dynamic output shape: {dynamic_output.shape}")
except Exception as e:
    print(f"Dynamic output shape: {str(e)}")

# Test 2: Static output shape (torch.compile)
print("\n2. Static output shape (torch.compile):")
try:
    compiled_func = torch.compile(call_func, dynamic=True)
    static_output = compiled_func(inputs, stride=stride, padding=padding, output_padding=output_padding, groups=groups, dilation=dilation)
    print(f"Static output shape: {static_output.shape}")
except Exception as e:
    print(f"Static output shape: {str(e)}")

# Test 3: Meta output shape (device='meta')
print("\n3. Meta output shape (device='meta'):")
try:
    meta_input_tensor = torch.randn(2, 32, 10, 10, 10, device='meta')
    meta_weight = torch.randn(32, 2, 3, 3, 3, device='meta')
    meta_inputs = [meta_input_tensor, meta_weight]
    
    meta_output = call_func(meta_inputs, stride=stride, padding=padding, output_padding=output_padding, groups=groups, dilation=dilation)
    print(f"Meta output shape: {list(meta_output.shape)}")
except Exception as e:
    print(f"Meta output shape: {str(e)}")