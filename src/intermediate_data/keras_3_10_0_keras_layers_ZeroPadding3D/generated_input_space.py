import numpy as np
from dataclasses import dataclass, field
from typing import Union, List, Tuple, Optional

# 1. Define valid_test_case variable
input_tensor = np.random.randn(2, 4, 6, 8, 3).astype(np.float32)
valid_test_case = {
    "padding": ((1, 1), (2, 2), (3, 3)),
    "data_format": None,
    "inputs": input_tensor
}

# 2. & 3. & 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    """Dataclass containing all parameters affecting ZeroPadding3D output shape"""
    
    padding: List[Union[int, Tuple[int, int, int], Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]] = field(
        default_factory=lambda: [
            0,  # int: symmetric padding 0
            2,  # int: symmetric padding 2
            (1, 2, 3),  # tuple of 3 ints
            ((1, 1), (2, 2), (3, 3)),  # tuple of 3 tuples of 2 ints
            ((0, 2), (1, 3), (2, 4))  # tuple of 3 tuples of 2 ints (asymmetric)
        ]
    )
    
    data_format: List[Optional[str]] = field(
        default_factory=lambda: [None, "channels_last", "channels_first"]
    )