import torch
from dataclasses import dataclass, field
from typing import Union, List, Tuple

# 1. Define valid_test_case
valid_test_case = {
    "padding": 2,
    "inputs": torch.randn(1, 1, 3, 3)
}

# 2 & 3. Identify parameters and define value spaces
# Only 'padding' affects output shape besides 'inputs'
# Padding can be int (same padding all sides) or 4-tuple (left, right, top, bottom)

@dataclass
class InputSpace:
    padding: List[Union[int, Tuple[int, int, int, int]]] = field(default_factory=lambda: [
        # Integer padding values (discrete)
        0,     # no padding
        1,     # minimal positive padding
        2,     # from example
        3,     # typical small value
        5,     # typical medium value
        10,    # typical larger value
        
        # 4-tuple padding values
        # Boundary/edge cases
        (0, 0, 0, 0),           # no padding
        (1, 1, 1, 1),           # uniform padding
        (1, 0, 0, 0),           # left only
        (0, 1, 0, 0),           # right only
        (0, 0, 1, 0),           # top only
        (0, 0, 0, 1),           # bottom only
        
        # Asymmetric cases
        (2, 2, 2, 2),           # from example (equivalent to int=2)
        (1, 1, 2, 0),           # from example
        (2, 3, 1, 4),           # different values on all sides
        (0, 5, 0, 5),           # horizontal padding only
        (3, 3, 0, 0),           # vertical padding only
        
        # Larger values
        (10, 10, 10, 10),       # large uniform padding
        (5, 0, 10, 0),          # mixed large/small values
    ])