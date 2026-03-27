import torch
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, List

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": [torch.randn(3, 4)],
    "dim": 1,
    "keepdim": True,
    "out": None
}

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    """
    Dataclass containing all parameters that affect the output shape of torch.any
    with discretized value ranges
    """
    dim: List[Optional[Union[int, Tuple[int]]]] = field(
        default_factory=lambda: [None, 0, 1, -1, (0, 1)]
    )
    keepdim: List[bool] = field(
        default_factory=lambda: [True, False]
    )