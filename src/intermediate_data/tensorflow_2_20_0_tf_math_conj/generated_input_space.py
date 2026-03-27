import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import List, Union, Optional

valid_test_case = {
    'inputs': tf.constant(np.random.randn(3, 2) + 1j * np.random.randn(3, 2), dtype=tf.complex64),
    'name': None
}

@dataclass
class InputSpace:
    # There are no parameters besides 'inputs' that affect output tensor shape
    # 'name' parameter exists but doesn't affect output shape
    pass