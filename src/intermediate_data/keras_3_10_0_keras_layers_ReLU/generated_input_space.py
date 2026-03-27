from dataclasses import dataclass
import numpy as np

# Task 1: Define valid_test_case
valid_test_case = {
    'max_value': 10,
    'negative_slope': 0.5,
    'threshold': 0,
    'name': 'relu_layer',
    'dtype': None,
    'inputs': np.random.randn(5, 3)
}

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    # No parameters affect the output tensor shape - all are scalar values
    # that only affect element-wise computations
    pass

# Note: The ReLU layer in Keras performs element-wise operations only.
# None of its parameters (max_value, negative_slope, threshold, name, dtype)
# affect the output tensor shape - the output always matches input shape.
# The dtype parameter affects data type but not shape.
# Therefore, InputSpace contains no fields.