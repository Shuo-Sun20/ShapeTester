import tensorflow as tf
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

# 1. Define valid_test_case with all parameters
valid_test_case = {
    "inputs": [tf.constant(np.random.randn(2, 3, 4).astype(np.float32))],
    "name": "test_transpose",  # Set a non-None name to avoid the error
    "conjugate": False
}

# 2. Parameters that can affect the shape of the output tensor (except inputs):
#    None - tf.linalg.matrix_transpose ONLY swaps the last two dimensions of input tensor.
#    Therefore, no parameters in call_func except "inputs" affect the output shape.

# 3. Since no parameters affect shape except inputs, InputSpace has no fields related to shape.
#    However, to satisfy the requirement of having a class that can be instantiated,
#    we include an empty dataclass.

# 4. Define InputSpace class
@dataclass
class InputSpace:
    # No parameters in call_func except "inputs" affect output tensor shape
    pass