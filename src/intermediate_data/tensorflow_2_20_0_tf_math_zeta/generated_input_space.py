import tensorflow as tf
import numpy as np
from dataclasses import dataclass

def call_func(inputs, name=None):
    x = inputs[0]
    q = inputs[1]
    return tf.math.zeta(x, q, name)

# Generate random tensors for x and q
np.random.seed(42)
x_tensor = tf.constant(np.random.rand(3, 3).astype(np.float32))
q_tensor = tf.constant(np.random.rand(3, 3).astype(np.float32))

# Task 1: Define valid_test_case
valid_test_case = {
    'inputs': [x_tensor, q_tensor]
}

# Task 2, 3, 4: Define InputSpace
@dataclass
class InputSpace:
    # No parameters other than 'inputs' affect output shape in call_func
    pass