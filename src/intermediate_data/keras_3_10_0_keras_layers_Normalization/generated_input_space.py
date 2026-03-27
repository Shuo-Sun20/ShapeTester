import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Union, List, Tuple, Optional

# 1. Valid test case
valid_test_case = {
    "inputs": np.array([[1., 2., 3.], [4., 5., 6.]], dtype='float32'),
    "axis": -1,
    "mean": None,
    "variance": None,
    "invert": False
}

# 2. & 3. Parameter analysis for output shape affecting parameters
# For Normalization layer, the output shape is always the same as input shape.
# However, parameters that affect broadcasting compatibility (and thus can cause errors) are:
# axis: Determines which axes are normalized, affects weight shapes
# mean: Must be broadcastable to input shape based on axis
# variance: Must be broadcastable to input shape based on axis
# invert: Doesn't affect shape but affects values

@dataclass
class InputSpace:
    # axis: Integer, tuple, or None - controls normalization axes
    axis: List[Union[int, Tuple[int, ...], None]] = field(
        default_factory=lambda: [
            None,   # Normalize all elements globally
            -1,     # Normalize last axis (default)
            0,      # Normalize first axis
            (0, 1), # Normalize first two axes
            (1,)    # Normalize second axis only
        ]
    )
    
    # mean: None or broadcastable array - affects normalization values
    mean: List[Optional[Union[float, np.ndarray]]] = field(
        default_factory=lambda: [
            None,  # Learn from data
            0.0,   # Zero mean
            1.0,   # Positive mean
            -1.0,  # Negative mean
            0.5    # Fractional mean
        ]
    )
    
    # variance: None or broadcastable array - affects normalization values
    variance: List[Optional[Union[float, np.ndarray]]] = field(
        default_factory=lambda: [
            None,  # Learn from data
            1.0,   # Unit variance
            0.1,   # Small variance
            10.0,  # Large variance
            0.01   # Very small variance
        ]
    )
    
    # invert: Boolean - affects transformation direction
    invert: List[bool] = field(
        default_factory=lambda: [False, True]
    )