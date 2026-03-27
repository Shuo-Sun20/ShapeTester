import keras
from dataclasses import dataclass

# 1. Define valid_test_case
x = keras.ops.convert_to_tensor([-6.0, 1.0, 0.0, 1.0, 6.0])
valid_test_case = {"inputs": x}

# 2. & 3. & 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # Note: Only 'inputs' parameter affects output shape, but it's excluded per requirements.
    # Therefore, InputSpace contains no fields as there are no other parameters.
    pass