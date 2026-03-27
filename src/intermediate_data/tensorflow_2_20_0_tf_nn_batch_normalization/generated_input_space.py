import tensorflow as tf
from dataclasses import dataclass

# 1. Define valid_test_case dictionary
valid_test_case = {
    'inputs': [
        tf.random.normal([4, 6]),                      # x: [batch, depth]
        tf.random.normal([6]),                         # mean: [depth]
        tf.abs(tf.random.normal([6])),                 # variance: [depth]
        tf.random.normal([6]),                         # offset: [depth]
        tf.random.normal([6])                          # scale: [depth]
    ],
    'variance_epsilon': 0.001
}

# 2. Parameters in call_func that can affect output shape: None
# 3. Only parameter besides inputs is variance_epsilon, which doesn't affect shape
#    It's a continuous float parameter that prevents division by zero

@dataclass
class InputSpace:
    # Only parameter besides inputs is variance_epsilon
    # Discretized value space for variance_epsilon:
    # Boundary values: very small positive float, reasonable range for typical use
    variance_epsilon: list = None
    
    def __post_init__(self):
        if self.variance_epsilon is None:
            # Include boundary values and typical values in ML
            # From very small (close to 0) to typical values like 1e-5, 1e-3, 0.01
            self.variance_epsilon = [
                1e-10,       # Very small boundary value
                1e-8,        # Very small
                1e-5,        # Common default in many frameworks
                0.001,       # Example from documentation
                0.01,        # Larger value
                0.1,         # Upper bound for typical use
                1.0          # Unusually large, but still valid
            ]