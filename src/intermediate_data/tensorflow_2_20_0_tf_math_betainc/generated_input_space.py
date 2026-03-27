import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field

# Step 1: Define valid_test_case dictionary
np.random.seed(42)
tensor_a = tf.constant(np.random.uniform(0.1, 5.0, (3, 3)), dtype=tf.float32)
tensor_b = tf.constant(np.random.uniform(0.1, 5.0, (3, 3)), dtype=tf.float32)
tensor_x = tf.constant(np.random.uniform(0.0, 1.0, (3, 3)), dtype=tf.float32)

valid_test_case = {
    'inputs': [tensor_a, tensor_b, tensor_x],
    'name': 'test_betainc'
}

# Step 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    # The only parameter in call_func that affects shape is 'inputs'
    # Since 'inputs' is a list of tensors, we need to consider the shapes of these tensors
    # We'll create example tensor shapes that can be broadcast together
    inputs: list = field(default_factory=lambda: [
        # Shape (1,): scalar
        [
            tf.constant([2.0], dtype=tf.float32),
            tf.constant([3.0], dtype=tf.float32),
            tf.constant([0.5], dtype=tf.float32)
        ],
        # Shape (3,): vector
        [
            tf.constant([1.0, 2.0, 3.0], dtype=tf.float32),
            tf.constant([4.0, 5.0, 6.0], dtype=tf.float32),
            tf.constant([0.1, 0.5, 0.9], dtype=tf.float32)
        ],
        # Shape (2, 2): matrix
        [
            tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32),
            tf.constant([[5.0, 6.0], [7.0, 8.0]], dtype=tf.float32),
            tf.constant([[0.1, 0.2], [0.3, 0.4]], dtype=tf.float32)
        ],
        # Shape (1, 3): row vector
        [
            tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32),
            tf.constant([[4.0, 5.0, 6.0]], dtype=tf.float32),
            tf.constant([[0.1, 0.5, 0.9]], dtype=tf.float32)
        ],
        # Shape (3, 1): column vector
        [
            tf.constant([[1.0], [2.0], [3.0]], dtype=tf.float32),
            tf.constant([[4.0], [5.0], [6.0]], dtype=tf.float32),
            tf.constant([[0.1], [0.5], [0.9]], dtype=tf.float32)
        ]
    ])
    
    # 'name' parameter doesn't affect shape but is in the signature
    name: list = field(default_factory=lambda: [
        None,
        'betainc_op_1',
        'betainc_op_2',
        'betainc_op_3',
        'custom_name'
    ])