import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Any, List, Union

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": np.random.randn(2, 4, 4, 3).astype(np.float32),
    "alpha_initializer": "Zeros",
    "alpha_regularizer": None,
    "alpha_constraint": None,
    "shared_axes": [1, 2],
    "name": "test_prelu",
    "dtype": "float32"
}

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    # Only shared_axes affects output shape due to parameter sharing
    # Possible values: None (no sharing), [1] (height), [2] (width), [1,2] (height+width)
    shared_axes: List[Optional[List[int]]] = field(default_factory=lambda: [
        None, [1], [2], [1,2], [2,3]
    ])