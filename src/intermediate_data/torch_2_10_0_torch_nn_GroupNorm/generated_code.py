import torch
import torch.nn as nn

def call_func(num_groups, num_channels, eps=1e-5, affine=True, inputs=None):
    gn = nn.GroupNorm(num_groups, num_channels, eps, affine)
    output = gn(inputs)
    return output

example_input = torch.randn(20, 6, 10, 10)
example_output = call_func(3, 6, inputs=example_input)