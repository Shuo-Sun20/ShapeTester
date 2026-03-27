import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import List

def call_func(inputs, name=None):
    return tf.math.real(inputs, name=name)

# 1. valid_test_case definition
np.random.seed(42)
real_part = np.random.randn(3, 4).astype(np.float32)
imag_part = np.random.randn(3, 4).astype(np.float32)
complex_tensor = tf.constant(real_part + 1j * imag_part)

valid_test_case = {
    'inputs': complex_tensor,
    'name': 'test_real_operation'
}

# 4. InputSpace definition
@dataclass
class InputSpace:
    inputs: List[tf.Tensor] = field(default_factory=lambda: [
        # Scalar complex
        tf.constant(2.0 + 3.0j, dtype=tf.complex64),
        # 1D complex tensor
        tf.constant([1.5 - 2.5j, -3.0 + 4.0j], dtype=tf.complex64),
        # 2D complex tensor
        tf.constant([[1.0 + 2.0j, 3.0 - 4.0j], [-5.0 + 6.0j, 7.0 + 8.0j]], dtype=tf.complex64),
        # 3D real tensor (real input)
        tf.constant(np.random.randn(2, 3, 4).astype(np.float32)),
        # 4D complex tensor
        tf.constant(np.random.randn(3, 2, 4, 1).astype(np.complex64))
    ])