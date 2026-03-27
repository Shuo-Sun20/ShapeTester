import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List

# 1. Define valid_test_case
np.random.seed(42)
tensor1 = np.random.rand(2, 3, 4).astype(np.float32)
tensor2 = np.random.rand(2, 3, 4).astype(np.float32)
valid_test_case = {
    "inputs": [tensor1, tensor2],
    "name": None
}

# 2-4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # The only parameter that affects output shape (other than inputs) is `name`,
    # but `name` doesn't affect the tensor shape. However, for completeness,
    # we include all parameters from call_func().
    name: List[str] = field(default_factory=lambda: [None, "subtract_layer", "custom_name", "layer_1", "layer_2"])

# The output shape is determined solely by the shape of the input tensors,
# which must be identical for both inputs in the Subtract layer.
# No other parameters in call_func() affect the output tensor shape.