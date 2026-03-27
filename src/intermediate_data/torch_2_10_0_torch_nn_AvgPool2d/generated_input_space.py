import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Union, Tuple, List, Optional

# 1. Define valid_test_case
valid_test_case = {
    "kernel_size": 3,
    "stride": 2,
    "padding": 1,
    "ceil_mode": False,
    "count_include_pad": True,
    "divisor_override": None,
    "inputs": [torch.randn(1, 3, 64, 64)]
}

# 2 & 3 & 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    """
    Dataclass containing all parameters that affect the output shape of AvgPool2d.
    Each parameter's value space is discretized with typical values covering legal scenarios.
    """
    # kernel_size: Union[int, Tuple[int, int]]
    kernel_size: List[Union[int, Tuple[int, int]]] = field(
        default_factory=lambda: [
            1, 2, 3, 5, 7,  # Single int values
            (2, 3), (3, 3), (5, 7), (1, 1), (7, 7)  # Tuple values
        ]
    )
    
    # stride: Optional[Union[int, Tuple[int, int]]]
    stride: List[Optional[Union[int, Tuple[int, int]]]] = field(
        default_factory=lambda: [
            None, 1, 2, 3, 5,  # Single int values (None means equal to kernel_size)
            (1, 2), (2, 3), (3, 3), (5, 1), (1, 1)  # Tuple values
        ]
    )
    
    # padding: Union[int, Tuple[int, int]]
    padding: List[Union[int, Tuple[int, int]]] = field(
        default_factory=lambda: [
            0, 1, 2, 3, 5,  # Single int values
            (0, 1), (1, 2), (2, 2), (3, 5), (0, 0)  # Tuple values
        ]
    )
    
    # ceil_mode: bool
    ceil_mode: List[bool] = field(
        default_factory=lambda: [True, False]
    )