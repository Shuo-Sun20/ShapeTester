import torch
import torch.nn as nn

def call_func(kernel_size, output_size=None, output_ratio=None, return_indices=False, inputs=None):
    module = nn.FractionalMaxPool2d(
        kernel_size=kernel_size,
        output_size=output_size,
        output_ratio=output_ratio,
        return_indices=return_indices
    )
    return module(inputs)

input_tensor = torch.randn(20, 16, 50, 32)
example_output = call_func(
    kernel_size=3,
    output_size=(13, 12),
    return_indices=False,
    inputs=input_tensor
)