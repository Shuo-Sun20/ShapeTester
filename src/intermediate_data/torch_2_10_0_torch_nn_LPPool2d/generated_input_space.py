import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Union, Tuple, List, Optional

# Task 1: Define valid_test_case
valid_test_case = {
    'norm_type': 2,
    'kernel_size': 3,
    'stride': 2,
    'ceil_mode': False,
    'inputs': torch.randn(20, 16, 50, 32)
}

# Task 2 & 3: Identify shape-affecting parameters and their value spaces
@dataclass
class InputSpace:
    """
    Defines parameter value spaces for torch.nn.LPPool2d parameters that affect output shape.
    
    Parameters:
        kernel_size: int or tuple of two ints. Must be positive.
        stride: int or tuple of two ints, or None. Must be positive.
        ceil_mode: bool.
    """
    # kernel_size can be int or tuple[int, int]
    kernel_size: List[Union[int, Tuple[int, int]]] = field(
        default_factory=lambda: [
            1, 2, 3, 4, 5,               # Single int values (boundary:1 and typical)
            (1, 1), (2, 2), (3, 3),      # Square tuples
            (1, 2), (2, 1), (3, 2),      # Non-square tuples
            (5, 1), (1, 5)               # Extreme aspect ratios
        ]
    )
    
    # stride can be int, tuple[int, int], or None (defaults to kernel_size)
    stride: List[Optional[Union[int, Tuple[int, int]]]] = field(
        default_factory=lambda: [
            None,                        # Default behavior
            1, 2, 3, 4, 5,               # Single int values
            (1, 1), (2, 2), (3, 3),      # Square tuples
            (1, 2), (2, 1), (3, 2),      # Non-square tuples
            (5, 1), (1, 5)               # Extreme aspect ratios
        ]
    )
    
    # ceil_mode is boolean
    ceil_mode: List[bool] = field(
        default_factory=lambda: [True, False]
    )