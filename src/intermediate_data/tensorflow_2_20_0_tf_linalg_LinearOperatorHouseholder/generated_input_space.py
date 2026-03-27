import tensorflow as tf
import numpy as np
from dataclasses import dataclass
from typing import List, Union

valid_test_case = {
    "reflection_axis": tf.constant([1.0, 0.0, 0.0, 0.0, 0.0], dtype=tf.float32),
    "is_non_singular": None,
    "is_self_adjoint": None,
    "is_positive_definite": None,
    "is_square": None,
    "name": "LinearOperatorHouseholder",
    "inputs": tf.constant(np.random.randn(5, 3), dtype=tf.float32)
}

@dataclass
class InputSpace:
    reflection_axis: List[Union[List[float], tf.Tensor]] = None
    
    def __post_init__(self):
        if self.reflection_axis is None:
            self.reflection_axis = [
                tf.constant([1.0, 0.0], dtype=tf.float32),
                tf.constant([1.0, 0.0, 0.0], dtype=tf.float32),
                tf.constant([1.0, 0.0, 0.0, 0.0], dtype=tf.float32),
                tf.constant([1.0, 0.0, 0.0, 0.0, 0.0], dtype=tf.float32),
                tf.constant([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=tf.float32)
            ]