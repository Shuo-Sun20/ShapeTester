import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union, List, Tuple, Dict, Any
from keras.initializers import Initializer
from keras.regularizers import Regularizer
from keras.constraints import Constraint

# Task 1: Define valid_test_case dictionary
valid_test_case = {
    "groups": 8,
    "axis": -1,
    "epsilon": 0.001,
    "center": True,
    "scale": True,
    "beta_initializer": "zeros",
    "gamma_initializer": "ones",
    "beta_regularizer": None,
    "gamma_regularizer": None,
    "beta_constraint": None,
    "gamma_constraint": None,
    "name": None,
    "dtype": None,
    "inputs": keras.ops.convert_to_tensor(np.random.randn(2, 32, 32, 64).astype(np.float32))
}

# Task 2 & 3: Parameters affecting output shape (excluding "inputs")
# Only 'axis' parameter affects output shape calculation

@dataclass
class InputSpace:
    axis: List[Union[int, List[int], Tuple[int]]] = field(
        default_factory=lambda: [
            -1,
            1,
            2,
            3,
            [1, 2, 3]
        ]
    )