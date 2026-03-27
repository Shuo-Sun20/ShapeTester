import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Union, List, Optional

# 1. Define valid_test_case with parameters for call_func
valid_test_case = {
    "inputs": keras.random.normal((2, 3, 4)),
    "shift": 1,
    "axis": 1
}

# 2. Parameters that can affect output shape (excluding "inputs"): 
# - axis (can be None, int, or tuple of ints - affects whether flattening occurs)
# - shift (does not affect shape - only affects content)
# Actually, only axis parameter affects whether flattening occurs during internal processing,
# but the output shape is always same as input shape. However, axis=None triggers flattening
# then restoration of original shape, so technically shape isn't affected.

def call_func(inputs, shift, axis=None):
    return keras.ops.roll(inputs, shift, axis)

# For completeness, we'll include both shift and axis in InputSpace since they both
# affect how the roll operation is performed, even if not affecting final shape

# 3. Parameter value space analysis:
# shift: int or tuple/list of ints (continuous/discrete)
# axis: None, int, tuple of ints (discrete)

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # shift: Can be int or tuple/list of ints
    # We'll create a list of representative values including boundary cases
    shift: List[Union[int, List[int], tuple]] = field(default_factory=lambda: [
        # Single integer cases
        0,  # no shift
        1,  # small positive
        -1,  # negative
        5,  # larger than dimension
        -5,  # negative larger than dimension
        
        # Tuple/list cases
        (1, 2),  # multiple shifts
        (-1, 2),  # mixed signs
        (0, 0, 0),  # all zeros
        (3, -3, 1),  # varied values
        
        # Boundary case from valid_test_case
        1,
    ])
    
    # axis: Can be None, int, or tuple/list of ints
    axis: List[Optional[Union[int, List[int], tuple]]] = field(default_factory=lambda: [
        None,  # flatten and restore
        0,  # first axis
        -1,  # last axis
        2,  # middle axis
        (0, 1),  # multiple axes
        (0, -1),  # mixed indexing
        (-2, -1),  # negative indices
        
        # Boundary cases
        1,  # from valid_test_case
    ])

# Example instantiation
var = InputSpace()