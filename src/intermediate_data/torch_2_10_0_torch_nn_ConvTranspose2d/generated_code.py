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

input_tensor = torch.randn(20, 16, 50, 100)
example_output = call_func(
    in_channels=16,
    out_channels=33,
    kernel_size=3,
    inputs=input_tensor,
    stride=2
)