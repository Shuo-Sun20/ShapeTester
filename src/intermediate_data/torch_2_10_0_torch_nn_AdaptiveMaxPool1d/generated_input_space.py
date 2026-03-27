import torch
from dataclasses import dataclass, field
from typing import Union, Tuple, List

# 1. Define a valid test case
valid_test_case = {
    "output_size": 5,
    "return_indices": False,
    "inputs": torch.randn(2, 3, 10)
}

# 2 & 3. Identify shape-affecting parameters and construct value spaces
# Parameters that affect output shape (excluding "inputs"): output_size

@dataclass
class InputSpace:
    """
    Data class containing all parameters that affect the output tensor shape
    of torch.nn.AdaptiveMaxPool1d through call_func(), with discretized value ranges.
    """
    # output_size can be int or tuple(int)
    output_size: List[Union[int, Tuple[int]]] = field(default_factory=lambda: [
        # Discrete positive integers (L_out > 0)
        1,     # Minimum valid value
        2,
        3,
        5,     # Value from valid_test_case
        7,
        10,    # Typical mid-value
        15,
        20,
        50,
        100,   # Large value
        # Single-element tuple format (also valid)
        (1,),
        (5,),
        (10,)
    ])