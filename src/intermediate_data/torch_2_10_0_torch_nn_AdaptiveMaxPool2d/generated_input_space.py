import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Union, Tuple, List, Optional

# 1. Define valid_test_case
torch.manual_seed(42)
input_tensor = torch.randn(1, 64, 8, 9)
valid_test_case = {
    'output_size': (5, 7),
    'inputs': input_tensor,
    'return_indices': False
}

# 2. Parameters affecting output shape (excluding "inputs"): output_size
# 3. Value space analysis for output_size:
#    Type: Union[int, Tuple[Optional[int], Optional[int]]]
#    Values can be:
#    - Single int (square output)
#    - Tuple with two ints (H_out, W_out)
#    - Tuple with None values (same as input dimension)
#    Note: output_size cannot be a single None

@dataclass
class InputSpace:
    # Parameter: output_size
    # Value space covering all legal scenarios:
    # 1. Single integer values (square output)
    # 2. Tuple of integers
    # 3. Tuple with None values (mixed cases)
    output_size: List[Union[int, Tuple[Optional[int], Optional[int]]]] = field(
        default_factory=lambda: [
            # Boundary values (single int)
            1,          # Minimum valid size
            32,         # Larger than typical input
            
            # Typical values (single int)
            4,
            7,
            10,
            14,
            
            # Tuple of ints
            (1, 1),     # Minimum 2D size
            (4, 4),     # Square
            (5, 7),     # Rectangle (valid_test_case value)
            (8, 9),     # Same as example input
            (10, 12),   # Rectangle
            
            # Tuple with None (mixed cases)
            (None, 7),  # Keep input height, specify width
            (5, None),  # Specify height, keep input width
            (None, None) # Keep both dimensions
        ]
    )

# Example instantiation
var = InputSpace()