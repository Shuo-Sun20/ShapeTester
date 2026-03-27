import torch
import torch.nn as nn

def call_func(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False, inputs=None):
    pool = nn.MaxPool3d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
    if isinstance(inputs, list):
        output = pool(*inputs)
    else:
        output = pool(inputs)
    return output

torch.manual_seed(0)
example_input = torch.randn(2, 3, 8, 8, 8)
example_output = call_func(kernel_size=2, stride=2, inputs=example_input)