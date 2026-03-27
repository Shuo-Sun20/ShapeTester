import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, List

# Valid test case
example_input = torch.randn(2, 3, 16, 16)
valid_test_case = {
    "inputs": example_input,
    "kernel_size": 2,
    "stride": 2,
    "padding": 0,
    "ceil_mode": False,
    "count_include_pad": True,
    "divisor_override": None
}

@dataclass
class InputSpace:
    """
    Dataclass containing all parameters that affect the output shape of F.avg_pool2d
    """
    # kernel_size can be int or tuple
    kernel_size: List[Union[int, Tuple[int, int]]] = field(
        default_factory=lambda: [
            1, 2, 3, 5, 7,  # Single integer values
            (1, 1), (2, 3), (3, 2), (3, 5), (5, 3), (7, 7)  # Tuple values
        ]
    )
    
    # stride can be None, int, or tuple
    stride: List[Optional[Union[int, Tuple[int, int]]]] = field(
        default_factory=lambda: [
            None,  # None means use kernel_size
            1, 2, 3, 4, 5,  # Single integer values
            (1, 1), (1, 2), (2, 1), (2, 3), (3, 2), (3, 5)  # Tuple values
        ]
    )
    
    # padding can be int or tuple
    padding: List[Union[int, Tuple[int, int]]] = field(
        default_factory=lambda: [
            0, 1, 2, 3, 4, 5,  # Single integer values
            (0, 0), (1, 1), (1, 2), (2, 1), (2, 3), (3, 2)  # Tuple values
        ]
    )
    
    # ceil_mode is boolean
    ceil_mode: List[bool] = field(
        default_factory=lambda: [True, False]
    )

# Example instantiation
var = InputSpace()