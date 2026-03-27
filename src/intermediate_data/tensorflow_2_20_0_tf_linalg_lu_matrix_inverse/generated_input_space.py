import tensorflow as tf
from dataclasses import dataclass, field

# Valid test case
x = tf.random.normal(shape=(3, 2, 2))
lu, perm = tf.linalg.lu(x)
valid_test_case = {
    'inputs': [lu, perm],
    'validate_args': False,
    'name': None
}

# Identify parameters affecting output shape
# Only parameters passed to tf.linalg.lu_matrix_inverse affect shape:
# - lower_upper (via inputs[0])
# - perm (via inputs[1])
# validate_args and name don't affect output shape

@dataclass
class InputSpace:
    """
    Contains parameters that could potentially affect output shape
    """
    validate_args: list = field(default_factory=lambda: [False, True])
    name: list = field(default_factory=lambda: [None, 'test_name'])