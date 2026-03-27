import torch
import torch.nn as nn

def call_func(output_size, inputs, return_indices=False):
    adaptive_max_pool = nn.AdaptiveMaxPool2d(output_size, return_indices)
    if return_indices:
        output, indices = adaptive_max_pool(inputs)
        return output
    else:
        output = adaptive_max_pool(inputs)
        return output

torch.manual_seed(42)
input_tensor = torch.randn(1, 64, 8, 9)
example_output = call_func((5, 7), input_tensor)