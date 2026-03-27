import numpy as np
from dataclasses import dataclass, field
from typing import Union, List, Optional, Tuple

# 1. Define valid_test_case with all call_func parameters
valid_test_case = {
    "pool_size": (2, 2),
    "strides": (2, 2),
    "padding": "valid",
    "data_format": None,
    "name": None,
    "kwargs": None,
    "inputs": np.random.randn(2, 8, 8, 3).astype(np.float32)
}

# 2 & 3 & 4. InputSpace dataclass containing shape-affecting parameters
@dataclass
class InputSpace:
    pool_size: List[Union[int, Tuple[int, int]]] = field(
        default_factory=lambda: [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
    )
    strides: List[Optional[Union[int, Tuple[int, int]]]] = field(
        default_factory=lambda: [None, (1, 1), (2, 2), (3, 3), (4, 4)]
    )
    padding: List[str] = field(
        default_factory=lambda: ["valid", "same"]
    )
    data_format: List[Optional[str]] = field(
        default_factory=lambda: [None, "channels_last", "channels_first"]
    )