import keras
import numpy as np

# Generate random boolean tensor for input (same as in example)
np.random.seed(42)
random_bool_array = np.random.choice([True, False], size=(3, 4, 5))
x = keras.ops.convert_to_tensor(random_bool_array)

# 1. Define valid_test_case
valid_test_case = {
    'inputs': x,
    'axis': 1,
    'keepdims': True
}

from dataclasses import dataclass, field

# 3 & 4. Define InputSpace dataclass with parameter value spaces
@dataclass
class InputSpace:
    axis: list = field(default_factory=lambda: [None, 0, 1, 2, (0, 1)])
    keepdims: list = field(default_factory=lambda: [True, False])