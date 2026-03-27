import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import List

# 1. Define valid_test_case
np.random.seed(42)
valid_test_case = {
    "inputs": [
        tf.constant(np.random.randn(3, 2).astype(np.float32)),
        tf.constant(np.random.rand(3, 2).astype(np.float32))
    ]
}

# 2. Parameters affecting output shape:
#    Only the tensors in 'inputs' (a and x) affect shape through broadcasting.
#    The 'name' parameter does not affect the output shape.

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    """Contains all parameters that affect the output shape, excluding 'inputs'."""
    
    # Note: There are no parameters in call_func() that affect output shape
    # (other than the tensors passed in 'inputs', which are excluded by instructions)
    pass

# The InputSpace class can be instantiated successfully
var = InputSpace()