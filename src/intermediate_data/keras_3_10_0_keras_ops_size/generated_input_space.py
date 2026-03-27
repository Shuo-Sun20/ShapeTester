import keras
import numpy as np
from dataclasses import dataclass, field

valid_test_case = {
    "inputs": keras.ops.convert_to_tensor(np.random.randn(3, 4, 5).astype(np.float32))
}

@dataclass
class InputSpace:
    # The `call_func` only has one parameter "inputs"
    # No other parameters exist that could affect the output shape
    pass