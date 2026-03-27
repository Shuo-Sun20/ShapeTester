import torch
import torch.nn as nn

def call_func(kernel_size, output_size=None, output_ratio=None, return_indices=False, inputs=None):
    pool = nn.FractionalMaxPool3d(
        kernel_size=kernel_size,
        output_size=output_size,
        output_ratio=output_ratio,
        return_indices=return_indices
    )
    
    output = pool(inputs)
    
    if return_indices:
        return output[0]
    else:
        return output

input_tensor = torch.randn(20, 16, 50, 32, 16)
example_output = call_func(
    kernel_size=3,
    output_size=(13, 12, 11),
    return_indices=False,
    inputs=input_tensor
)