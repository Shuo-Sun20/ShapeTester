import keras
import numpy as np
from dataclasses import dataclass, field

def call_func(inputs, k, sorted=True):
    values, indices = keras.ops.top_k(inputs, k, sorted)
    return [values, indices]

random_tensor = keras.random.normal(shape=(10,))

# 1. valid_test_case
valid_test_case = {
    "inputs": random_tensor,
    "k": 3,
    "sorted": True
}

# 2. Parameters affecting output shape: k
# 3. Value space analysis:
#    k: discrete integer parameter (positive, <= input size)
#       values: [1, 2, 3, 4, 5, 10] (covering min, typical, max for shape (10,))
#    sorted: boolean parameter (discrete)
#       values: [True, False]

# 4. InputSpace dataclass
@dataclass
class InputSpace:
    k: list = field(default_factory=lambda: [1, 2, 3, 4, 5, 10])
    # Note: 'sorted' parameter is intentionally omitted as it doesn't affect output shape