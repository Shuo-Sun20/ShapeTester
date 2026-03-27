import torch
import torch.nn.functional as F

def call_func(inputs, kernel_size, output_size=None, output_ratio=None, return_indices=False):
    if isinstance(inputs, list):
        if len(inputs) == 1:
            input_tensor = inputs[0]
            random_samples = None
        elif len(inputs) == 2:
            input_tensor = inputs[0]
            random_samples = inputs[1]
        else:
            raise ValueError("inputs list must contain 1 or 2 tensors")
    else:
        raise TypeError("inputs must be a list of tensors")
    
    return F.fractional_max_pool3d(
        input=input_tensor,
        kernel_size=kernel_size,
        output_size=output_size,
        output_ratio=output_ratio,
        return_indices=return_indices,
        _random_samples=random_samples
    )

input_tensor = torch.randn(20, 16, 50, 32, 16)
example_output = call_func([input_tensor], kernel_size=3, output_size=(13, 12, 11))