import torch
from typing import Tuple, List, Union
from dataclasses import dataclass, field
import math

# 1. Define valid_test_case
valid_test_case = {
    "padding": (3, 3, 6, 6, 1, 1),
    "inputs": [torch.randn(16, 3, 8, 320, 480)]
}

# 2. Parameters affecting output shape (excluding "inputs"): "padding"
# 3. Value space analysis for padding:
#    Type: Union[int, Tuple[int, int, int, int, int, int]]
#    Discrete values for int padding: 0 (boundary), 1, 3, 10, 100 (typical)
#    For tuple padding: each element can be 0 (boundary), 1, 3, 10, 100
#    We'll create representative tuples covering symmetric/asymmetric cases

@dataclass
class InputSpace:
    padding: List[Union[int, Tuple[int, int, int, int, int, int]]] = field(
        default_factory=lambda: [
            # Integer padding values
            0, 1, 3, 10, 100,
            # Tuple padding values
            (0, 0, 0, 0, 0, 0),  # No padding
            (1, 1, 1, 1, 1, 1),  # Symmetric small padding
            (3, 3, 3, 3, 3, 3),  # Symmetric medium padding
            (10, 10, 10, 10, 10, 10),  # Symmetric large padding
            (3, 3, 6, 6, 1, 1),  # Asymmetric padding from valid_test_case
            (0, 5, 0, 5, 0, 5),  # Asymmetric only right/bottom/back
            (5, 0, 5, 0, 5, 0),  # Asymmetric only left/top/front
            (2, 4, 6, 8, 10, 12),  # Fully asymmetric padding
            (100, 100, 100, 100, 100, 100),  # Large symmetric padding
        ]
    )