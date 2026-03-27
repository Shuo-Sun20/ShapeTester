import torch
import numpy as np
import torch
import torch.nn as nn

def call_func(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int],
    inputs: torch.Tensor,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    output_padding: int | tuple[int, int] = 0,
    groups: int = 1,
    bias: bool = True,
    dilation: int | tuple[int, int] = 1,
    output_size: tuple[int, int] | None = None
) -> torch.Tensor:
    conv_transpose = nn.ConvTranspose2d(
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
    
    if output_size is not None:
        output = conv_transpose(inputs, output_size)
    else:
        output = conv_transpose(inputs)
    
    return output

# Test parameters that cause the defect
in_channels = 32
out_channels = 256
kernel_size = 5
stride = [1, 2]
padding = 1
output_padding = [1, 0]
groups = 1
bias = True
dilation = 1
output_size = None

# Create input tensor with mismatched channels (16 instead of 32)
inputs = torch.randn(1, 16, 10, 10)

print("Testing dynamic execution:")
try:
    dynamic_output = call_func(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        inputs=inputs,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        bias=bias,
        dilation=dilation,
        output_size=output_size
    )
    print(f"Dynamic output shape: {dynamic_output.shape}")
except Exception as e:
    print(f"Dynamic output shape: {e}")

print("\nTesting compiled execution:")
try:
    compiled_func = torch.compile(call_func, dynamic=True)
    static_output = compiled_func(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        inputs=inputs,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        bias=bias,
        dilation=dilation,
        output_size=output_size
    )
    print(f"Static output shape: {static_output.shape}")
except Exception as e:
    print(f"Static output shape: {e}")

print("\nTesting meta device execution:")
try:
    inputs_meta = torch.randn(1, 16, 10, 10, device='meta')
    meta_output = call_func(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        inputs=inputs_meta,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        bias=bias,
        dilation=dilation,
        output_size=output_size
    )
    print(f"Meta output shape: {meta_output.shape}")
except Exception as e:
    print(f"Meta output shape: {e}")