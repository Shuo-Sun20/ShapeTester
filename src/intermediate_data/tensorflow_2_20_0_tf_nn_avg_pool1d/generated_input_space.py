import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import List, Union

# 1. Define valid_test_case
valid_test_case = {
    'inputs': tf.convert_to_tensor(np.random.randn(4, 10, 3).astype(np.float32)),
    'ksize': 2,
    'strides': 2,
    'padding': 'VALID',
    'data_format': 'NWC',
    'name': None
}

# 2. & 3. & 4. Define InputSpace
@dataclass
class InputSpace:
    ksize: List[Union[int, List[int]]] = field(
        default_factory=lambda: [1, 2, 3, 4, [1, 2, 1]]
    )
    strides: List[Union[int, List[int]]] = field(
        default_factory=lambda: [1, 2, 3, 4, [1, 2, 1]]
    )
    padding: List[str] = field(default_factory=lambda: ['VALID', 'SAME'])
    data_format: List[str] = field(default_factory=lambda: ['NWC', 'NCW'])