import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Union, List, Tuple, Optional

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

# 1. Define valid_test_case dictionary
valid_test_case = {
    "inputs": [input_tensor],
    "kernel_size": 3,
    "output_size": (13, 12, 11),
    "output_ratio": None,
    "return_indices": False
}

# 2. Parameters affecting output shape: kernel_size, output_size, output_ratio
# 3. Discretized value spaces:

@dataclass
class InputSpace:
    """Contains all parameters affecting output tensor shape with discretized value ranges."""
    
    # kernel_size can be int or tuple of 3 ints
    kernel_size: List[Union[int, Tuple[int, int, int]]] = field(
        default_factory=lambda: [
            # Single integer (square kernel)
            1, 2, 3, 5, 7,
            # Tuple of 3 ints
            (1, 1, 1), (2, 2, 2), (3, 3, 3), (2, 3, 4), (3, 5, 7)
        ]
    )
    
    # output_size can be None, int, or tuple of 3 ints
    output_size: List[Optional[Union[int, Tuple[int, int, int]]]] = field(
        default_factory=lambda: [
            None,  # When using output_ratio
            # Single integer (cubic output)
            10, 13, 16, 20, 25,
            # Tuple of 3 ints (boundary and typical values)
            (10, 10, 10), (13, 12, 11), (16, 16, 16), (20, 16, 10), (25, 20, 15)
        ]
    )
    
    # output_ratio can be None, float, or tuple of 3 floats in (0, 1)
    output_ratio: List[Optional[Union[float, Tuple[float, float, float]]]] = field(
        default_factory=lambda: [
            None,  # When using output_size
            # Single float (cubic ratio)
            0.1, 0.25, 0.5, 0.75, 0.9,
            # Tuple of 3 floats
            (0.1, 0.1, 0.1), (0.25, 0.5, 0.75), (0.5, 0.5, 0.5), 
            (0.75, 0.5, 0.25), (0.9, 0.9, 0.9)
        ]
    )