import keras
import numpy as np
from dataclasses import dataclass

# 1. Valid test case
real_part = keras.random.normal(shape=(3, 4))
imag_part = keras.random.normal(shape=(3, 4))
complex_np = np.array(real_part) + 1j * np.array(imag_part)
complex_tensor = keras.ops.convert_to_tensor(complex_np)
valid_test_case = {"inputs": [complex_tensor]}

# 2, 3, 4. InputSpace dataclass definition
@dataclass
class InputSpace:
    # Note: The conjugate operation doesn't have additional parameters that affect output shape
    # beyond the input tensor itself. The only parameter is 'inputs', which is excluded per task requirements.
    # Therefore, there are no additional parameters to define in InputSpace.
    pass