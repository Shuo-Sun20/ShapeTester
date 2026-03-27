import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List

# 1. Define valid_test_case
batch_size = 3
num_classes = 4
target_tensor = keras.random.randint(shape=(batch_size,), minval=0, maxval=num_classes)
output_tensor = keras.random.uniform(shape=(batch_size, num_classes))
valid_test_case = {
    "inputs": [target_tensor, output_tensor],
    "from_logits": False,
    "axis": -1
}

# 2 & 3 & 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    """
    Contains all parameters of call_func (except 'inputs') that can affect
    the shape of the output tensor, with discretized value spaces.
    """
    axis: List[int] = field(default_factory=lambda: [-5, -3, -2, -1, 0, 1, 2, 3, 4])
    # Note: from_logits does NOT affect output shape, but is included per specification
    # that InputSpace contains **all** parameters affecting shape (axis is the only one).
    # We include from_logits for completeness of the function's parameter list.
    from_logits: List[bool] = field(default_factory=lambda: [False, True])

# Test instantiation
var = InputSpace()