import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field

def call_func(inputs, pooling_ratio, pseudo_random=False, overlapping=False, seed=0, name=None):
    return tf.nn.fractional_avg_pool(value=inputs, pooling_ratio=pooling_ratio, pseudo_random=pseudo_random, overlapping=overlapping, seed=seed, name=name)[0]

# 1. Define valid_test_case
example_input = tf.constant(np.random.randn(4, 10, 10, 3).astype(np.float32))
valid_test_case = {
    "inputs": example_input,
    "pooling_ratio": [1.0, 1.44, 1.73, 1.0],
    "pseudo_random": False,
    "overlapping": False,
    "seed": 0,
    "name": None
}

# 2. Identify shape-affecting parameters (except inputs)
# Only pooling_ratio affects the output tensor shape
# pseudo_random, overlapping, seed, and name do NOT affect output shape

# 3. Value space analysis for pooling_ratio
# pooling_ratio: list of floats with length >=4, first and last must be 1.0
# Height ratio (index 1) and width ratio (index 2) are continuous parameters >= 1.0
# For height and width ratios, we discretize the value space with boundary values and 5 typical values

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    pooling_ratio: list = field(default_factory=lambda: [
        [1.0, 1.0, 1.0, 1.0],           # No pooling (boundary)
        [1.0, 1.2, 1.2, 1.0],           # Typical value 1
        [1.0, 1.44, 1.44, 1.0],         # Typical value 2 (from example)
        [1.0, 1.73, 1.73, 1.0],         # Typical value 3 (from example)
        [1.0, 2.0, 2.0, 1.0],           # Typical value 4
        [1.0, 2.5, 2.5, 1.0],           # Typical value 5
        [1.0, 3.0, 3.0, 1.0],           # Larger ratio
        [1.0, 1.44, 1.73, 1.0],         # Different H and W ratios
        [1.0, 1.73, 1.44, 1.0],         # Different H and W ratios (reversed)
        [1.0, 1.0, 2.0, 1.0],           # Only width pooling
        [1.0, 2.0, 1.0, 1.0]            # Only height pooling
    ])