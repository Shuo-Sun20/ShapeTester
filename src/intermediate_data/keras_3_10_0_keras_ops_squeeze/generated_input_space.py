import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple

def call_func(inputs, axis=None):
    x = inputs[0]
    return keras.ops.squeeze(x, axis)

# Construct a random tensor with shape (1, 3, 1, 5)
x = keras.random.normal(shape=(1, 3, 1, 5))
example_output = call_func(inputs=[x])

# 1. Define valid_test_case
valid_test_case = {
    "inputs": [x],  # must be a list containing at least one tensor
    "axis": None  # valid axis parameter
}

# 2 & 3. Parameter analysis for axis:
# Type: Union[None, int, Tuple[int, ...]] 
# Value space considerations:
# - None: Remove all size-1 dimensions
# - int: Single axis index (positive or negative)
# - Tuple[int]: Multiple axis indices
# For the example tensor (1, 3, 1, 5):
# Valid axes: 0, 2, -3, (0, 2), (0, -3), (2, 0), (-3, 0), etc.

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    """Class containing all parameters affecting output shape except 'inputs'"""
    axis: list = field(default_factory=lambda: [
        # Discrete value space covering all legal scenarios
        None,  # Remove all size-1 dims
        0,     # Positive single axis
        2,     # Positive single axis
        -3,    # Negative single axis (same as 2 for shape (1,3,1,5))
        -1,    # Invalid axis (dimension 5, not size-1) - should raise error
        (0, 2),     # Tuple of valid axes
        (0, -3),    # Mixed positive/negative
        (2, 0),     # Different order
        (-3, 0),    # Negative first
        (0, 2, -3), # Redundant tuple (0 and -3 same axis)
        tuple(),    # Empty tuple
        (0, 1),     # Invalid (axis 1 has size 3)
        (0, 2, 4),  # Invalid (axis 4 out of range)
    ])

# Test instantiation
var = InputSpace()