import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, List

# Task 1: Define valid_test_case variable
valid_test_case = {
    'inputs': keras.random.uniform(shape=(3, 4)) > 0.5,
    'axis': 1,
    'keepdims': True
}

def call_func(inputs, axis=None, keepdims=False):
    return keras.ops.any(x=inputs, axis=axis, keepdims=keepdims)

# Task 2 & 3: Parameters affecting output shape: axis and keepdims

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    """
    Dataclass containing all parameters that affect the shape of keras.ops.any output.
    
    Parameters:
    - axis: Can be None, int, or tuple of ints. For 2D input (3,4):
        * None: reduces all dimensions
        * Single axis: 0, 1, -1, -2
        * Multiple axes: (0,1), (1,0), (-1,-2), (0,), (1,)
    - keepdims: Boolean parameter
    """
    axis: List[Optional[Union[int, Tuple[int, ...]]]] = field(
        default_factory=lambda: [
            None,           # Reduce all dimensions
            0,              # Reduce along rows
            1,              # Reduce along columns
            -1,             # Reduce along last axis (columns)
            -2,             # Reduce along second-to-last axis (rows)
            (0, 1),         # Reduce both rows and columns (all dimensions)
            (1, 0),         # Same as (0, 1) but different order
            (0,),           # Reduce rows only (tuple form)
            (1,),           # Reduce columns only (tuple form)
            (-1, -2),       # Reduce both axes (negative indices)
            (-2, -1),       # Same as (-1, -2) but different order
            (0,),           # Boundary: single axis as tuple
            (1,),           # Boundary: single axis as tuple
        ]
    )
    
    keepdims: List[bool] = field(
        default_factory=lambda: [True, False]
    )

# This ensures InputSpace can be instantiated with var=InputSpace()
if __name__ == "__main__":
    # Test instantiation
    space = InputSpace()
    print(f"InputSpace instantiated successfully")
    print(f"axis values: {space.axis}")
    print(f"keepdims values: {space.keepdims}")