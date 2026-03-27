from dataclasses import dataclass, field
from typing import Union, Tuple, List
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

# 1. Valid test case dictionary
valid_test_case = {
    "output_size": (5, 7, 9),
    "return_indices": False,
    "inputs": torch.randn(1, 64, 8, 9, 10)
}

# 2. Parameters affecting output shape (except "inputs"): output_size

# 3. Parameter type analysis and value space:
#    output_size: Union[int, Tuple[Union[int, None], Union[int, None], Union[int, None]]]
#    Legal values include:
#    - Single int (for cubic output)
#    - Tuple of 3 ints
#    - Tuple of 3 values where each can be int or None
#    All int values must be positive (>0)

# 4. InputSpace dataclass definition
@dataclass
class InputSpace:
    # Discrete parameter: output_size
    # Value space includes all legal value scenarios with at least 5 typical values
    output_size: List[Union[int, Tuple[Union[int, None], Union[int, None], Union[int, None]]]] = field(
        default_factory=lambda: [
            # Single int (cubic outputs) - boundary and typical values
            1,           # Minimum valid size
            2,           # Small size
            5,           # Typical small-medium size (included from valid_test_case as part of tuple)
            7,           # Medium size
            10,          # Medium-large size
            
            # Tuple of 3 ints - various scenarios
            (1, 1, 1),   # Minimum 3D size
            (5, 7, 9),   # Valid test case value
            (3, 5, 7),   # Different dimensions
            (8, 8, 8),   # Cubic from tuple
            (10, 10, 10),# Large cubic
            
            # Tuple with None values - boundary and typical cases
            (None, None, None),      # Keep all dimensions same
            (5, None, None),         # Change only depth
            (None, 7, None),         # Change only height
            (None, None, 9),         # Change only width
            (5, None, 9),            # Change depth and width
            (None, 7, 9),            # Change height and width
            (5, 7, None),            # Change depth and height
        ]
    )

# Example instantiation
var = InputSpace()