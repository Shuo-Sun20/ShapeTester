from dataclasses import dataclass, field
from typing import Union, Optional, Tuple, List
import torch

# 1. Define valid_test_case
valid_test_case = {
    'kernel_size': (2, 2, 2),
    'stride': (2, 2, 2),
    'padding': 0,
    'ceil_mode': False,
    'count_include_pad': True,
    'divisor_override': None,
    'inputs': torch.randn(2, 3, 8, 16, 16)
}

# 2 & 3. Define InputSpace dataclass with all parameters affecting output shape
@dataclass
class InputSpace:
    """Dataclass containing parameters affecting AvgPool3d output shape with discretized value spaces"""
    
    kernel_size: List[Union[int, Tuple[int, int, int]]] = field(
        default_factory=lambda: [
            # Single int values
            1, 2, 3, 4, 5,
            # Tuple values
            (2, 2, 2),  # From valid_test_case
            (3, 3, 3), (1, 2, 3), (4, 4, 4), (5, 5, 5),
            # Boundary values
            (1, 1, 1), (8, 8, 8)
        ]
    )
    
    stride: List[Optional[Union[int, Tuple[int, int, int]]]] = field(
        default_factory=lambda: [
            # None (defaults to kernel_size)
            None,
            # Single int values
            1, 2, 3, 4,
            # Tuple values
            (2, 2, 2),  # From valid_test_case
            (1, 1, 1), (3, 3, 3), (2, 1, 2), (1, 2, 1),
            # Boundary values
            (8, 8, 8)
        ]
    )
    
    padding: List[Union[int, Tuple[int, int, int]]] = field(
        default_factory=lambda: [
            # Single int values
            0,  # From valid_test_case
            1, 2, 3, 4,
            # Tuple values
            (0, 0, 0), (1, 1, 1), (2, 2, 2), (1, 2, 3), (3, 2, 1),
            # Boundary values
            8
        ]
    )
    
    ceil_mode: List[bool] = field(
        default_factory=lambda: [
            False,  # From valid_test_case
            True
        ]
    )

# 4. Ensure the class can be instantiated
var = InputSpace()