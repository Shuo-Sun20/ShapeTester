import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union, List, Tuple

# 1. Define valid_test_case
valid_test_case = {
    "inputs": [keras.ops.convert_to_tensor(np.random.randn(3, 4))],
    "axis": 1,
    "keepdims": False
}

# 2 and 3. Define InputSpace dataclass with parameters that affect output shape
@dataclass
class InputSpace:
    """
    Data class containing parameters that affect the output shape of keras.ops.amin.
    """
    axis: List[Optional[Union[int, Tuple[int, ...]]]] = field(
        default_factory=lambda: [None, 0, 1, (0, 1), -1]
    )
    keepdims: List[bool] = field(default_factory=lambda: [True, False])