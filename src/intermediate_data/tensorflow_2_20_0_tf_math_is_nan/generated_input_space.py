import tensorflow as tf
import numpy as np
from dataclasses import dataclass

valid_test_case = {
    'inputs': tf.constant([1.0, np.nan, 3.14, float('nan'), float('inf')], dtype=tf.float32),
    'name': None
}

@dataclass
class InputSpace:
    name: list[str | None] = None
    
    def __post_init__(self):
        if self.name is None:
            self.name = [None, '', 'isnan_op', 'test_is_nan', 'tf_math_isnan']