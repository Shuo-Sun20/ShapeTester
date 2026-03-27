import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Union, Optional, List

valid_test_case = {
    "inputs": keras.ops.convert_to_tensor(np.random.randn(3, 4, 5)),
    "dtype": "float32"
}

def call_func(inputs, dtype=None):
    return keras.ops.zeros_like(x=inputs, dtype=dtype)

@dataclass
class InputSpace:
    dtype: List[Optional[Union[str, type]]] = field(default_factory=lambda: [
        None,  # default (uses input tensor's dtype)
        "float32",
        "float64",
        "int32",
        "int64",
        "bool"
    ])