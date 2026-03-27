import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Union, List, Tuple, Optional

def call_func(inputs, axis=None, keepdims=False):
    return keras.ops.std(inputs, axis=axis, keepdims=keepdims)

example_tensor = keras.ops.convert_to_tensor(np.random.randn(3, 4))
valid_test_case = {'inputs': example_tensor, 'axis': 1, 'keepdims': True}

@dataclass
class InputSpace:
    # Parameters that affect output shape (excluding 'inputs')
    axis: List[Union[int, Tuple[int, ...], None]] = field(default_factory=lambda: [
        None,           # Reduce all axes (flattened)
        0,              # Reduce along axis 0 (rows)
        1,              # Reduce along axis 1 (columns)
        -1,             # Reduce along last axis
        (0, 1),         # Reduce along multiple axes
        (1, 0),         # Same axes, different order
        (0,),           # Single-element tuple for axis 0
        (-1, 0),        # Mixed positive/negative indices
        (-2, -1),       # Negative indices only
    ])
    keepdims: List[bool] = field(default_factory=lambda: [False, True])