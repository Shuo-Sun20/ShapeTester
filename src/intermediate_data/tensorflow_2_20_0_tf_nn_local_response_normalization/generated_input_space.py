import tensorflow as tf
from dataclasses import dataclass, field
from typing import List

def call_func(inputs, depth_radius=5, bias=1.0, alpha=1.0, beta=0.5, name=None):
    output = tf.nn.local_response_normalization(
        input=inputs,
        depth_radius=depth_radius,
        bias=bias,
        alpha=alpha,
        beta=beta,
        name=name
    )
    return output

# Generate random 4D tensor as input (batch, height, width, channels)
input_tensor = tf.random.normal(shape=(2, 4, 4, 3), dtype=tf.float32)

# 1. Define valid_test_case dictionary
valid_test_case = {
    'inputs': input_tensor,
    'depth_radius': 5,
    'bias': 1.0,
    'alpha': 1.0,
    'beta': 0.5,
    'name': None
}

# 2. Parameters that affect output shape (none - all parameters only affect values, not shape)
# However, we'll include all computation parameters as requested

# 3. Value spaces for each parameter (max 5 values each)

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # Note: None of these parameters affect output shape - output always matches input shape
    depth_radius: List[int] = field(default_factory=lambda: [0, 1, 3, 5, 7])
    bias: List[float] = field(default_factory=lambda: [0.1, 0.5, 1.0, 2.0, 5.0])
    alpha: List[float] = field(default_factory=lambda: [0.0001, 0.001, 0.01, 0.1, 1.0])
    beta: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0, 1.5])