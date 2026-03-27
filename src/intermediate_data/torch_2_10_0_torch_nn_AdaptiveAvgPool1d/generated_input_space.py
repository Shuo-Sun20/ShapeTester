import torch
from typing import Union, List, Tuple
from dataclasses import dataclass, field
from typing import Any

# 1. Valid test case definition
valid_test_case = {
    'output_size': 5,
    'inputs': torch.randn(2, 3, 10)
}

# 2. Parameters affecting output shape (except inputs): only output_size

# 3. Value space analysis for output_size
# output_size must be positive integer ≥ 1
# It can be:
# - An integer (e.g., 5)
# - A single-element tuple (e.g., (5,))
# Values are discrete but infinite, so we discretize to representative values

# 4. InputSpace class definition
@dataclass
class InputSpace:
    output_size: List[Union[int, Tuple[int]]] = field(
        default_factory=lambda: [
            # Boundary and small values
            1,  # Minimum valid value
            2,  # Small value
            
            # Typical values
            4,  # Less than typical input length
            5,  # Example from valid_test_case
            
            # Equal to common input lengths
            8,   # Equal to common input size
            10,  # Equal to common input size
            
            # Larger than input length (still valid)
            15,  # Greater than input
            20,  # Much greater than input
            
            # Tuple form (equivalent to integer)
            (5,),   # Tuple version of example
            (10,),  # Tuple version
            
            # Very large but valid value
            1000
        ]
    )
    
    def __post_init__(self):
        # Ensure all values are valid (positive integers or tuples)
        valid_values = []
        for val in self.output_size:
            if isinstance(val, int):
                if val >= 1:
                    valid_values.append(val)
            elif isinstance(val, tuple) and len(val) == 1 and isinstance(val[0], int) and val[0] >= 1:
                valid_values.append(val)
        self.output_size = valid_values