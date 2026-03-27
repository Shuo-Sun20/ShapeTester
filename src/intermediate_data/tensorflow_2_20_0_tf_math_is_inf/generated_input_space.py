import tensorflow as tf
import numpy as np
from dataclasses import dataclass

valid_test_case = {
    'inputs': tf.constant([1.0, np.inf, -np.inf, 5.0, np.nan]),
    'name': None
}

@dataclass
class InputSpace:
    name: list = None
    
    def __post_init__(self):
        if self.name is None:
            self.name = [None, '', 'is_inf_op', 'a' * 10, 'A' * 100]