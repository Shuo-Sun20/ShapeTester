import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List, Union

def call_func(inputs, axes, keepdims=False, synchronized=False):
    mean, variance = keras.ops.moments(
        x=inputs,
        axes=axes,
        keepdims=keepdims,
        synchronized=synchronized
    )
    return [mean, variance]

# Generate random input tensor
np.random.seed(42)
x = np.random.randn(3, 4, 5).astype("float32")

# 1. Define valid_test_case
valid_test_case = {
    "inputs": x,
    "axes": [1],
    "keepdims": True,
    "synchronized": False
}

# 2. Parameters affecting output shape (except "inputs"): axes, keepdims

# 3. Value space analysis:
#    - axes: Discrete parameter (list/tuple of integers). Needs to handle:
#        * Single axis (int or list with one element)
#        * Multiple axes (list/tuple)
#        * Negative indices
#        * Empty list (reduce over all dimensions)
#        * Boundaries: [-rank, rank-1]
#    - keepdims: Boolean parameter with values [True, False]

# 4. InputSpace dataclass definition
@dataclass
class InputSpace:
    # For 3D input tensor (shape: (3,4,5)), rank=3
    # axes value space: 5+ typical values covering all legal scenarios
    axes: List[Union[int, List[int], tuple]] = field(default_factory=lambda: [
        [0],                    # Single positive axis
        [1],                    # Another single axis (from valid_test_case)
        [2],                    # Last axis
        [-1],                   # Single negative axis
        [-3],                   # Negative boundary (maps to axis 0)
        [0, 1],                 # Two consecutive axes
        [0, 2],                 # Two non-consecutive axes
        [0, 1, 2],              # All axes explicitly
        [],                     # Empty list (reduce over all dimensions)
        (0,),                   # Tuple instead of list
        [-2, -1]                # Two negative axes
    ])
    
    keepdims: List[bool] = field(default_factory=lambda: [True, False])