import torch
import torch.nn.functional as F

def call_func(inputs, kernel_size, output_size=None, output_ratio=None, _random_samples=None):
    if isinstance(inputs, list):
        input_tensor = inputs[0]
        if len(inputs) > 1:
            _random_samples = inputs[1]
    else:
        input_tensor = inputs
    
    output, _ = F.fractional_max_pool3d_with_indices(
        input_tensor,
        kernel_size=kernel_size,
        output_size=output_size,
        output_ratio=output_ratio,
        _random_samples=_random_samples
    )
    return output

input_tensor = torch.randn(20, 16, 50, 32, 16)
example_output = call_func([input_tensor], 3, output_size=(13, 12, 11))