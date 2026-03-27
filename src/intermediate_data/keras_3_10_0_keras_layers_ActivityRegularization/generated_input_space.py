import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List, Any

# 1. Define valid_test_case
valid_test_case = {
    "l1": 0.01,
    "l2": 0.02,
    "inputs": np.random.randn(2, 5, 8).astype('float32')
}

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    """
    Dataclass containing all parameters that can affect the output tensor shape.
    Since ActivityRegularization doesn't change output shape, this class has no fields.
    """
    # No parameters affect output shape for ActivityRegularization layer
    # The output shape is always identical to input shape
    pass

# Note: The ActivityRegularization layer does NOT change the output tensor shape.
# The output shape is always identical to the input shape, regardless of l1/l2 values.
# Therefore, no parameters (except "inputs") affect the output shape.