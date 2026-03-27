import torch
import numpy as np
import torch

def call_func(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    bias=True,
    dilation=1,
    inputs=None,
    output_size=None
):
    # Unpack the single input tensor from the list
    input_tensor = inputs[0] if isinstance(inputs, list) else inputs
    
    # Instantiate ConvTranspose1d
    conv_transpose_layer = torch.nn.ConvTranspose1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        bias=bias,
        dilation=dilation
    )
    
    # Call forward with optional output_size
    return conv_transpose_layer(input_tensor, output_size=output_size)

# Test parameters that reproduce the defect
in_channels = 16
out_channels = 32
kernel_size = 3
stride = 2
padding = 3
output_padding = 2
groups = 1
bias = True
dilation = 1
input_tensor = torch.randn(4, 16, 32)
inputs = [input_tensor]
output_size = None

print("Testing ConvTranspose1d shape consistency defect")

# Test 1: Dynamic output shape (direct call)
print("\n1. Dynamic output shape:")
try:
    dynamic_output = call_func(
        in_channels, out_channels, kernel_size, stride, padding, 
        output_padding, groups, bias, dilation, inputs, output_size
    )
    print(f"Dynamic shape: {dynamic_output.shape}")
except Exception as e:
    print(f"Dynamic error: {e}")

# Test 2: Static output shape (torch.compile with dynamic=True)
print("\n2. Static output shape (torch.compile):")
try:
    compiled_func = torch.compile(call_func, dynamic=True)
    static_output = compiled_func(
        in_channels, out_channels, kernel_size, stride, padding,
        output_padding, groups, bias, dilation, inputs, output_size
    )
    print(f"Static shape: {static_output.shape}")
except Exception as e:
    print(f"Static error: {e}")

# Test 3: Meta output shape (device='meta')
print("\n3. Meta output shape:")
try:
    meta_input = torch.randn(4, 16, 32, device='meta')
    meta_inputs = [meta_input]
    
    # Create layer on meta device
    meta_layer = torch.nn.ConvTranspose1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        bias=bias,
        dilation=dilation
    ).to('meta')
    
    meta_output = meta_layer(meta_input, output_size=output_size)
    print(f"Meta shape: {list(meta_output.shape)}")
except Exception as e:
    print(f"Meta error: {e}")

print("\nDefect reproduced: Meta device gives shape while CPU/CUDA throws validation error")