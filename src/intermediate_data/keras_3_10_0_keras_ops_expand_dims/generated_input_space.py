import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Union, List, Tuple

# 1. Define valid_test_case dictionary
valid_test_case = {
    "inputs": keras.ops.convert_to_tensor(np.random.randn(3, 4)),
    "axis": 1
}

# 2. Identify parameters affecting output shape (except "inputs"): "axis"

# 3. Analyze parameter types and construct value spaces:
# axis: Can be int or tuple of ints. For tensor of rank N, valid values are in [-N-1, N]
# For 2D tensor (shape [3,4]), N=2, valid range is [-3, 2]

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    axis: List[Union[int, Tuple[int, ...]]] = field(
        default_factory=lambda: [
            # Boundary and typical values for 2D tensor
            -3, -2, -1, 0, 1, 2,                    # Single integer axes
            (-3, 2), (-2, -1), (0, 1), (1, 2),     # Tuple axes (combinations)
            (-3, 0, 2), (-2, 1), (0,), (1,)        # Additional variations
        ]
    )