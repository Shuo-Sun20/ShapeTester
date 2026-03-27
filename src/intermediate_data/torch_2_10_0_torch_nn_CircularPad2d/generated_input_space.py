import torch
from torch.nn import CircularPad2d
from dataclasses import dataclass
from typing import Union, List, Tuple

def call_func(padding, inputs):
    pad_layer = CircularPad2d(padding)
    return pad_layer(inputs)

# 1. Define valid_test_case
valid_test_case = {
    "padding": 2,
    "inputs": torch.randn(1, 3, 32, 32)
}

# 2. Identify parameters affecting output shape (except inputs)
# Only "padding" affects the output shape

# 3. Analyze parameter types and construct value space for padding
# Padding can be: int or tuple of 4 ints
# For int: must be >= -min(H, W) and <= min(H, W) for typical 32x32 input
# For tuple: each element must satisfy same constraint relative to corresponding dimension

# Define typical input dimensions for value space construction
H, W = 32, 32

# For int padding: discrete values covering negative, zero, and positive ranges
int_padding_values = [
    -H,  # maximum negative padding (removes entire dimension)
    -16, # typical negative padding
    -8,  # moderate negative padding
    -1,  # minimal negative padding
    0,   # no padding
    1,   # minimal positive padding
    2,   # example from documentation
    8,   # moderate positive padding
    16,  # large positive padding
    H    # maximum positive padding (full wrap-around)
]

# For tuple padding: generate representative tuples
tuple_padding_values = []
# Symmetric padding cases
for p in [0, 1, 2, 8, 16, 32]:
    tuple_padding_values.append((p, p, p, p))
# Asymmetric padding cases (left, right, top, bottom)
tuple_padding_values.extend([
    (1, 2, 3, 4),      # all different positive
    (-1, -2, -3, -4),  # all different negative
    (1, -1, 2, -2),    # mixed positive/negative
    (0, H, 0, H),      # maximum wrap on right/bottom
    (H, 0, H, 0),      # maximum wrap on left/top
    (1, 1, 2, 0),      # example from documentation
    (16, 0, 0, 16),    # asymmetric large padding
])

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # padding can be either int or tuple of 4 ints
    padding: List[Union[int, Tuple[int, int, int, int]]] = None
    
    def __post_init__(self):
        if self.padding is None:
            # Combine all possible padding values
            self.padding = int_padding_values + tuple_padding_values

# Example instantiation
var = InputSpace()