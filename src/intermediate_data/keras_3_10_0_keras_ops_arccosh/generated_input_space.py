import keras
from dataclasses import dataclass
from typing import List

# 1. Define valid_test_case
valid_test_case = {
    "inputs": keras.ops.convert_to_tensor([10, 100])
}

# 2 & 3. Only "inputs" parameter affects output shape, no other parameters exist

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # Since call_func has only one parameter "inputs" which affects output shape,
    # we need to define its value space. However, note that "inputs" is excluded
    # from InputSpace per task requirements. Therefore, InputSpace is empty.
    pass