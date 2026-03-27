import keras
from dataclasses import dataclass

# 1. Define valid_test_case
valid_test_case = {
    'inputs': keras.random.uniform(shape=(3,))
}

# 2, 3, 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # There are no parameters in call_func besides 'inputs' that affect the output shape.
    pass