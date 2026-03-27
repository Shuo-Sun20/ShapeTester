import tensorflow as tf
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Union

# Task 1: Define valid_test_case
input_tensor = tf.constant(np.random.randn(4, 3).astype(np.float32))
valid_test_case = {
    'inputs': [input_tensor],
    'rcond': None,
    'validate_args': False,
    'name': 'pinv'
}

# Task 2 & 3: Parameters affecting output shape (only rcond affects shape)
# Task 4: InputSpace dataclass
@dataclass
class InputSpace:
    rcond: Optional[List[Union[float, None]]] = None
    
    def __post_init__(self):
        if self.rcond is None:
            self.rcond = [None, 1e-15, 1e-10, 1e-5, 1e-3]