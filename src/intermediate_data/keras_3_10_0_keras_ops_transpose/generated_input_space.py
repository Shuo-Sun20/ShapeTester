import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

def call_func(inputs, axes=None):
    return keras.ops.transpose(x=inputs, axes=axes)

# 1. Define valid_test_case
example_input = keras.random.normal(shape=(2, 3, 4))
valid_test_case = {
    "inputs": example_input,
    "axes": None
}

# 2. Parameters affecting output shape (excluding "inputs"): axes

# 3. Analysis of axes parameter:
# Type: Optional[Tuple[int, ...]] or Optional[List[int]]
# Possible values: 
# - None: reverses axes order
# - Tuple/List of integers: permutation of input tensor dimension indices

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # axes parameter: can be None or a permutation tuple/list
    # For a typical 3D input tensor (shape: 2,3,4), the valid permutations are:
    # - None (equivalent to (2, 1, 0) for 3D)
    # - All permutations of (0, 1, 2): 6 possible permutations
    
    # Generate all permutations for 3D tensor (0, 1, 2)
    PERMUTATIONS_3D = [
        None,  # Default case
        (0, 1, 2),  # Original order
        (0, 2, 1),  # Swap last two dimensions
        (1, 0, 2),  # Swap first two dimensions
        (1, 2, 0),  # Rotate dimensions
        (2, 0, 1),  # Rotate dimensions
        (2, 1, 0),  # Reverse order (same as default)
    ]
    
    # Boundary and typical values for different tensor ranks (1D to 4D)
    # We include permutations for tensors of different ranks
    
    # For 1D tensor - only one valid permutation
    PERMUTATIONS_1D = [
        None,
        (0,),
        (-1,),  # Negative indexing
    ]
    
    # For 2D tensor - 2 possible permutations
    PERMUTATIONS_2D = [
        None,
        (0, 1),
        (1, 0),
        (-1, -2),  # Negative indexing
        (-2, -1),  # Reverse with negative indices
        (0, -1),   # Mixed positive and negative
    ]
    
    # For 4D tensor - 24 possible permutations, we select representative ones
    PERMUTATIONS_4D = [
        None,
        (0, 1, 2, 3),  # Original
        (3, 2, 1, 0),  # Reverse
        (0, 1, 3, 2),  # Swap last two
        (1, 0, 2, 3),  # Swap first two
        (0, 2, 3, 1),  # Reorder
        (2, 3, 0, 1),  # Swap block
        (-1, -2, -3, -4),  # Negative indices
        (0, -1, 1, 2),     # Mixed indices
    ]
    
    # Combine all representative values for different tensor ranks
    axes: List[Optional[Tuple[int, ...]]] = field(
        default_factory=lambda: [
            # Boundary values
            None,  # Default case
            
            # 1D tensor cases (already included in PERMUTATIONS_1D)
            (0,),
            (-1,),
            
            # 2D tensor cases
            (0, 1),
            (1, 0),
            (-1, -2),
            
            # 3D tensor cases - from PERMUTATIONS_3D
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
            (-1, -2, -3),
            (0, -1, 1),
            
            # 4D tensor cases
            (0, 1, 2, 3),
            (3, 2, 1, 0),
            (0, 1, 3, 2),
            (1, 0, 2, 3),
            (0, 2, 3, 1),
            (2, 3, 0, 1),
            (-1, -2, -3, -4),
            (0, -1, 1, 2),
        ]
    )

# Example instantiation
var = InputSpace()