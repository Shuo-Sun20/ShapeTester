import torch
import torch.nn as nn

def call_func(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False, inputs=None):
    pool = nn.MaxPool1d(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        return_indices=return_indices,
        ceil_mode=ceil_mode
    )
    if return_indices:
        output, indices = pool(inputs)
        return output, indices
    else:
        output = pool(inputs)
        return output

example_input = torch.randn(20, 16, 50)
example_output = call_func(kernel_size=3, stride=2, inputs=example_input)