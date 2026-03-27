import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union, List

# 1. valid_test_case definition
x = keras.random.uniform(shape=(3, 4))
valid_test_case = {
    "inputs": x,
    "axis": 0,
    "dtype": 'float32'
}

# 2 & 3. Parameters affecting output shape: Only 'axis'
# Axis parameter type: Optional[int] for n-D tensors (can be negative)
# For a 2D tensor example (shape (3,4)):
# - None: flatten (output shape: (12,))
# - 0 or -2: cumulative sum along rows (output shape: (3,4))
# - 1 or -1: cumulative sum along columns (output shape: (3,4))
# Value space includes: None, valid positive axes (0,1), valid negative axes (-2,-1)

# 4. InputSpace dataclass definition
@dataclass
class InputSpace:
    axis: List[Optional[int]] = field(
        default_factory=lambda: [None, -2, -1, 0, 1]
    )
    # Note: 'inputs' parameter is not included as it's the base tensor
    # whose shape is handled separately in test construction