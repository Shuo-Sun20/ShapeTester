import torch
from dataclasses import dataclass, field
from typing import Union, List

# 1. Define valid_test_case
valid_test_case = {
    "padding": 2,
    "value": 3.5,
    "inputs": torch.randn(1, 2, 2)
}

# 2. Identify parameters that affect output tensor shape (except "inputs")
# Parameters: padding (affects shape), value (does not affect shape)
# So only "padding" affects shape

# 3. Value space analysis for padding parameter
# Padding can be: int (same padding on all sides) or 4-tuple (left, right, top, bottom)
# Both int and tuple values must be non-negative integers (>=0)
# Boundary values: 0 (no padding), and typical small positive integers
# Include the value from valid_test_case (2)

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    padding: List[Union[int, tuple]] = field(default_factory=lambda: [
        # int values
        0,  # boundary: no padding
        1,  # typical small value
        2,  # from valid_test_case
        3,  # typical small value
        5,  # typical small value
        10,  # typical larger value
        
        # tuple values
        (0, 0, 0, 0),  # boundary: no padding
        (1, 1, 1, 1),  # symmetric padding
        (2, 2, 2, 2),  # from valid_test_case (as tuple equivalent)
        (3, 0, 2, 1),  # example from documentation
        (0, 5, 0, 5),  # horizontal only padding
        (3, 3, 0, 0),  # vertical only padding
        (1, 2, 3, 4),  # asymmetric padding
        (10, 0, 0, 10),  # large asymmetric padding
    ])