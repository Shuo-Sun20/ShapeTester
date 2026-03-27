import torch
from dataclasses import dataclass, field
from typing import Union, List, Tuple

# 1. Define valid_test_case
valid_test_case = {
    'padding': (1, 2),
    'inputs': torch.randn(2, 8)
}

# 2. Parameters affecting output shape (except inputs): padding

# 3. Parameter type analysis and value space construction:
#    - padding: Union[int, Tuple[int, int]]
#      * Positive int: [0, 1, 2, 3, 4, 5] (max value depends on W_in, we assume W_in=8)
#      * Negative int: [-5, -4, -3, -2, -1] (bound by |padding| <= W_in)
#      * Tuple cases:
#        (0,0), (2,2) - symmetric
#        (1,3), (3,1) - asymmetric
#        (4,-2), (-2,4) - mixed signs
#        (-3,-3) - negative symmetric
#        (1,2) - from valid_test_case

# 4. InputSpace class definition
@dataclass
class InputSpace:
    padding: List[Union[int, Tuple[int, int]]] = field(
        default_factory=lambda: [
            # Positive int padding values (discretized)
            0, 1, 2, 3, 4, 5,
            # Negative int padding values (discretized)
            -5, -4, -3, -2, -1,
            # Tuple padding cases
            (0, 0),     # zero padding
            (2, 2),     # symmetric positive
            (1, 3),     # asymmetric positive
            (3, 1),     # asymmetric positive (reversed)
            (4, -2),    # mixed positive/negative
            (-2, 4),    # mixed negative/positive
            (-3, -3),   # symmetric negative
            (1, 2)      # from valid_test_case
        ]
    )

# This ensures InputSpace can be instantiated
var = InputSpace()