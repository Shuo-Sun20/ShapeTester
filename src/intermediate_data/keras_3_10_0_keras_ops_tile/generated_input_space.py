import keras
from dataclasses import dataclass, field
from typing import Union, Tuple, List

def call_func(inputs, repeats):
    return keras.ops.tile(inputs, repeats)

# 1. Valid test case
x = keras.random.normal(shape=(2, 3))
valid_test_case = {
    "inputs": x,
    "repeats": (2, 3)
}

# 2. Parameters affecting output shape (except "inputs"): "repeats"

# 3. Parameter analysis:
# repeats: Can be int, tuple of ints, list of ints, or tensor-like
# We'll use Union type for flexibility
# Discretized value space covering boundary and typical values:
# - Different dimensionalities (lengths)
# - Zero values (boundary)
# - Positive integers
# - Mixed combinations

# 4. InputSpace dataclass
@dataclass
class InputSpace:
    repeats: List[Union[int, Tuple[int, ...], List[int]]] = field(
        default_factory=lambda: [
            # Single integer repeats (1D case)
            0,  # boundary: no repetition
            1,  # boundary: single repetition
            2,  # typical: double repetition
            5,  # typical: multiple repetitions
            
            # 1D tuple/list
            (0,),  # boundary
            (1,),  # boundary
            (3,),  # typical
            
            # 2D tuples/lists
            (0, 0),  # boundary: zero in both dimensions
            (0, 1),  # boundary: zero in first dimension
            (1, 0),  # boundary: zero in second dimension
            (1, 1),  # boundary: single repetition in both
            (2, 3),  # from valid test case
            (1, 5),  # mixed
            (3, 2),  # reversed order
            
            # 3D tuples/lists (testing higher dimensionality)
            (1, 1, 1),  # boundary: single repetition in 3D
            (2, 1, 3),  # typical 3D pattern
            (1, 2, 1),  # mixed 3D
            
            # 4D case (testing even higher dimensionality)
            (1, 1, 1, 1),  # boundary
            (2, 3, 1, 2),  # typical 4D
            
            # List equivalents (testing different input types)
            [2, 3],  # list version of valid test case
            [0, 5],  # list with boundary
            [1, 2, 3],  # 3D list
            
            # Mixed values including zero
            (0, 3, 1),
            (2, 0, 2),
        ]
    )