import keras
from dataclasses import dataclass, field
from typing import Optional, Union, List, Tuple

# Valid test case from the example
example_tensor = keras.random.normal(shape=(3, 4))
valid_test_case = {
    "inputs": example_tensor,
    "axis": 1,
    "keepdims": True,
    "dtype": "float32"
}

@dataclass
class InputSpace:
    # Parameters affecting output shape
    axis: List[Optional[Union[int, Tuple[int, ...], List[int]]]] = field(
        default_factory=lambda: [
            None,           # Reduce all dimensions
            0,              # Reduce first dimension
            1,              # Reduce second dimension
            -1,             # Reduce last dimension
            -2,             # Reduce second-to-last dimension
            (0, 1),         # Reduce first two dimensions
            (0, -1),        # Reduce first and last dimensions
            (-1, -2),       # Reduce last two dimensions
            [0, 1],         # List version
            [-1, -2],       # List with negative indices
        ]
    )
    keepdims: List[bool] = field(
        default_factory=lambda: [True, False]
    )

# Instantiation example
var = InputSpace()