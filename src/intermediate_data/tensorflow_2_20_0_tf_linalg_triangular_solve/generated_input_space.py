import tensorflow as tf
from dataclasses import dataclass

# Recreate the example data for the valid test case
tf.random.set_seed(42)
matrix = tf.linalg.band_part(tf.random.normal((4, 4), dtype=tf.float32), -1, 0)
rhs = tf.random.normal((4, 2), dtype=tf.float32)

valid_test_case = {
    'inputs': [matrix, rhs],
    'lower': True,
    'adjoint': False,
    'name': None
}

# Define the InputSpace dataclass.
# Since no parameters (except inputs) affect the shape of the output, the class is empty.
@dataclass
class InputSpace:
    pass