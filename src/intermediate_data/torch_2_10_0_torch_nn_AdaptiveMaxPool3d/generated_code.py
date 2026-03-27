import torch
import torch.nn as nn

def call_func(output_size, return_indices, inputs):
    """
    Calls torch.nn.AdaptiveMaxPool3d with the given parameters.

    Parameters:
    - output_size: target output size (int or tuple)
    - return_indices: whether to return indices (bool)
    - inputs: input tensor (single tensor for AdaptiveMaxPool3d)

    Returns:
    - output tensor
    """
    adaptive_max_pool = nn.AdaptiveMaxPool3d(output_size, return_indices=return_indices)
    
    if return_indices:
        output, indices = adaptive_max_pool(inputs)
        return output
    else:
        output = adaptive_max_pool(inputs)
        return output

# Create random input tensor matching the example shape: (batch, channels, depth, height, width)
input_tensor = torch.randn(1, 64, 8, 9, 10)
example_output = call_func(output_size=(5, 7, 9), return_indices=False, inputs=input_tensor)