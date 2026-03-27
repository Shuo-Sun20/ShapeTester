import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Valid test case dictionary
valid_test_case = {
    'num_rows': 2,
    'num_columns': None,
    'batch_shape': None,
    'dtype': tf.float32,
    'is_non_singular': False,
    'is_self_adjoint': True,
    'is_positive_definite': False,
    'is_square': None,
    'name': None,
    'inputs': tf.random.normal(shape=[2, 4])
}

@dataclass
class InputSpace:
    num_rows: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    num_columns: List[Optional[int]] = field(default_factory=lambda: [None, 1, 2, 3, 4])
    batch_shape: List[Optional[Tuple[int, ...]]] = field(default_factory=lambda: [None, (1,), (2,), (2, 3), (1, 2, 3)])