import tensorflow as tf
import numpy as np
from dataclasses import dataclass

valid_test_case = {
    'inputs': [
        tf.constant(np.random.randn(3, 4).astype(np.float32)),
        tf.constant(np.random.randn(3, 4).astype(np.float32))
    ],
    'name': None
}

@dataclass
class InputSpace:
    """
    InputSpace class containing parameters that affect the shape of tf.math.floordiv output.
    The shape is determined by broadcasting rules of x and y tensors from 'inputs'.
    Since 'inputs' is excluded from consideration, only 'name' remains which does NOT affect shape.
    Therefore, no fields are defined as no parameters (except inputs) affect the output shape.
    """
    pass