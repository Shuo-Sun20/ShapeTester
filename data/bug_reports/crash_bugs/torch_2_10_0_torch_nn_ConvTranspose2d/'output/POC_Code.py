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

# Test input that causes the defect
inputs = torch.randn(1, 16, 11, 11)

# Test with regular execution (dynamic)
try:
    dynamic_output = call_func(
        in_channels=16,
        out_channels=64,
        kernel_size=7,
        stride=[1, 2],
        padding=[1, 0],
        output_padding=3,
        groups=2,
        bias=True,
        dilation=[1, 2],
        output_size=None,
        inputs=inputs
    )
    print(f"Dynamic output shape: {dynamic_output.shape}")
except Exception as e:
    print(f"Dynamic output shape: {e}")

# Test with torch.compile (static)
compiled_func = torch.compile(call_func, dynamic=True)
try:
    static_output = compiled_func(
        in_channels=16,
        out_channels=64,
        kernel_size=7,
        stride=[1, 2],
        padding=[1, 0],
        output_padding=3,
        groups=2,
        bias=True,
        dilation=[1, 2],
        output_size=None,
        inputs=inputs
    )
    print(f"Static output shape: {static_output.shape}")
except Exception as e:
    print(f"Static output shape: {e}")

# Test with meta device
meta_inputs = torch.randn(1, 16, 11, 11, device='meta')
try:
    meta_output = call_func(
        in_channels=16,
        out_channels=64,
        kernel_size=7,
        stride=[1, 2],
        padding=[1, 0],
        output_padding=3,
        groups=2,
        bias=True,
        dilation=[1, 2],
        output_size=None,
        inputs=meta_inputs
    )
    print(f"Meta output shape: {meta_output.shape}")
except Exception as e:
    print(f"Meta output shape: {e}")