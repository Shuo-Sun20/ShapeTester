import keras
import numpy as np
from dataclasses import dataclass

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": keras.ops.convert_to_tensor(
        np.random.randint(0, 100, size=(3, 4), dtype=np.int32)
    )
}

# Task 2: Parameters affecting output shape (excluding 'inputs')
# The bitwise_invert function only takes `x` as input parameter through call_func.
# Since `x` (provided as `inputs`) is the only parameter, there are no additional
# parameters in call_func that affect output shape beyond the tensor itself.
# Therefore, no parameters to list here.

# Task 3 & 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    # Since no parameters affect shape besides the tensor `inputs`,
    # we define an empty dataclass that can be instantiated.
    pass

# Example instantiation
var = InputSpace()