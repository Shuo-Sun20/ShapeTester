import keras
import numpy as np
from dataclasses import dataclass

# 1. Define valid_test_case dictionary
valid_test_case = {
    "inputs": keras.ops.convert_to_tensor(np.random.randn(3, 4) * 10.0)
}

# 2 & 3: No parameters other than "inputs" in call_func() that affect output shape
# The only parameter is "inputs", which is excluded per instructions

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # No parameters besides "inputs" that affect output shape
    # Therefore, define an empty dataclass with no fields
    pass