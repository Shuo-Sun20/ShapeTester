import torch
import torch.nn.functional as F

def call_func(
    inputs,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None
):
    return F.avg_pool2d(
        inputs,
        kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override
    )

example_input = torch.randn(2, 3, 16, 16)
example_output = call_func(example_input, kernel_size=2, stride=2, padding=0)