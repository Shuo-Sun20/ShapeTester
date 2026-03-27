import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Union, Tuple, Optional

# Keep the original function for completeness
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

# 1. Valid test case
valid_test_case = {
    'in_channels': 16,
    'out_channels': 32,
    'kernel_size': 3,
    'inputs': torch.randn(1, 16, 10, 10),
    'stride': 2,
    'padding': 1,
    'output_padding': 1,
    'groups': 1,
    'bias': True,
    'dilation': 1,
    'output_size': None
}

# 2 & 3. Parameters affecting output shape (except 'inputs') and their discretized value spaces
@dataclass
class InputSpace:
    # Parameters affecting output shape (spatial dimensions H_out, W_out)
    out_channels: list[int] = field(default_factory=lambda: [1, 16, 32, 64, 128, 256])
    kernel_size: list[Union[int, Tuple[int, int]]] = field(
        default_factory=lambda: [1, 2, 3, 5, 7, (2, 3), (3, 5), (5, 7)]
    )
    stride: list[Union[int, Tuple[int, int]]] = field(
        default_factory=lambda: [1, 2, 3, 4, 5, (1, 2), (2, 1), (2, 3)]
    )
    padding: list[Union[int, Tuple[int, int]]] = field(
        default_factory=lambda: [0, 1, 2, 3, 5, (0, 1), (1, 0), (1, 2)]
    )
    output_padding: list[Union[int, Tuple[int, int]]] = field(
        default_factory=lambda: [0, 1, 2, 3, (0, 1), (1, 0), (1, 2)]
    )
    dilation: list[Union[int, Tuple[int, int]]] = field(
        default_factory=lambda: [1, 2, 3, 4, 5, (1, 2), (2, 1), (2, 3)]
    )
    # Groups affects whether shape calculation is valid (output_padding constraints)
    groups: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    # output_size can override the formula-based shape calculation
    output_size: list[Optional[Tuple[int, int]]] = field(
        default_factory=lambda: [None, (10, 10), (16, 16), (24, 24), (32, 32), (48, 48)]
    )