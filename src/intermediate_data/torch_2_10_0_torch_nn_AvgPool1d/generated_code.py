import torch
import torch.nn as nn

def call_func(kernel_size, stride, padding, ceil_mode, count_include_pad, inputs):
    avgpool = nn.AvgPool1d(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad
    )
    output = avgpool(inputs)
    return output

torch.manual_seed(42)
example_input = torch.randn(2, 3, 10)
example_output = call_func(
    kernel_size=3,
    stride=2,
    padding=1,
    ceil_mode=False,
    count_include_pad=False,
    inputs=example_input
)