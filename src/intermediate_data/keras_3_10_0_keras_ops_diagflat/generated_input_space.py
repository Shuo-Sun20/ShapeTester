import keras
from dataclasses import dataclass, field

# 1. Valid test case dictionary
valid_test_case = {
    "inputs": keras.random.normal(shape=(4,)),
    "k": 0
}

# 2. Parameters affecting output shape (except "inputs"): k

# 3. Value space for parameter k (integer)
k_values = [-5, -3, -2, -1, 0, 1, 2, 3, 5]

# 4. InputSpace dataclass definition
@dataclass
class InputSpace:
    k: list = field(default_factory=lambda: k_values)