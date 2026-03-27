import numpy as np
import keras
from dataclasses import dataclass, field
from typing import List, Union, Tuple

def call_func(inputs, pad_width, mode="constant", constant_values=None):
    return keras.ops.pad(x=inputs, pad_width=pad_width, mode=mode, constant_values=constant_values)

# Valid test case as requested
valid_test_case = {
    "inputs": keras.ops.convert_to_tensor(np.random.randn(3, 4, 5)),
    "pad_width": ((1, 2), (3, 4), (5, 6)),
    "mode": "constant",
    "constant_values": 0.5
}

@dataclass
class InputSpace:
    pad_width: List[Union[int, Tuple[int, int], Tuple[Tuple[int, int], ...]]] = field(
        default_factory=lambda: [
            # Integer format (same padding for all axes)
            0,
            1,
            3,
            5,
            10,
            
            # Tuple of one integer (same before/after for all axes)
            (0,),
            (2,),
            (4,),
            
            # Tuple of two integers (same (before, after) for all axes)
            (1, 1),
            (2, 3),
            (3, 0),
            (0, 5),
            
            # Nested tuple format for 3D tensor (matching example)
            ((0, 0), (0, 0), (0, 0)),
            ((1, 1), (1, 1), (1, 1)),
            ((2, 3), (4, 5), (6, 7)),
            ((1, 2), (3, 4), (5, 6)),  # This is from valid_test_case
            ((5, 0), (0, 5), (3, 2)),
            ((10, 10), (10, 10), (10, 10)),
            
            # Asymmetric padding cases
            ((1, 0), (0, 1), (2, 2)),
            ((3, 1), (2, 4), (0, 3))
        ]
    )