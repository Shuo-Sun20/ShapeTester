import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Union, List, Tuple, Optional

def call_func(inputs, q, axis=None, method="linear", keepdims=False):
    return keras.ops.quantile(x=inputs, q=q, axis=axis, method=method, keepdims=keepdims)

valid_test_case = {
    "inputs": keras.random.uniform(shape=(5, 4, 3), minval=0.0, maxval=1.0),
    "q": [0.25, 0.5, 0.75],
    "axis": 1,
    "method": "midpoint",
    "keepdims": True
}

@dataclass
class InputSpace:
    """Discretized parameter space for call_func (excluding 'inputs' and 'method')."""
    
    q: List[Union[float, List[float]]] = field(
        default_factory=lambda: [
            0.0,  # boundary
            0.1,  # typical value
            0.25, # typical value
            0.5,  # typical value (median)
            0.75, # typical value
            0.9,  # typical value
            1.0,  # boundary
            [0.0, 0.5, 1.0],          # multiple probabilities
            [0.25, 0.5, 0.75],        # multiple probabilities (from valid_test_case)
            [0.1, 0.9]                # multiple probabilities
        ]
    )
    
    axis: List[Optional[Union[int, Tuple[int, ...]]]] = field(
        default_factory=lambda: [
            None,   # default (flatten)
            0,      # reduce along first axis
            1,      # reduce along second axis
            2,      # reduce along third axis
            -1,     # negative index
            (0, 1), # reduce along multiple axes
            (1, 2), # reduce along multiple axes
            (0, 2), # reduce along multiple axes
            (0, 1, 2) # reduce along all axes
        ]
    )
    
    keepdims: List[bool] = field(
        default_factory=lambda: [True, False]
    )

# Example instantiation
var = InputSpace()