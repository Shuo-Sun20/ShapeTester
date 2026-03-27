import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import List, Union

def call_func(inputs, ids, max_norm=None, name=None):
    params = inputs
    result = tf.nn.embedding_lookup(params, ids, max_norm=max_norm, name=name)
    return result

# Generate valid test case
np.random.seed(42)
params_tensor = np.random.randn(5, 3).astype(np.float32)
ids_tensor = np.array([0, 2, 4], dtype=np.int32)

valid_test_case = {
    'inputs': params_tensor,
    'ids': ids_tensor,
    'max_norm': None,
    'name': None
}

# Parameters that affect output shape: ids
@dataclass
class InputSpace:
    # ids affects shape through its tensor shape
    ids: List[Union[np.ndarray, tf.Tensor]] = field(default_factory=lambda: [
        np.array(0, dtype=np.int32),                    # scalar
        np.array([1, 3], dtype=np.int32),               # 1D, 2 elements
        np.array([[0, 2], [1, 3]], dtype=np.int32),     # 2D, 2x2
        np.array([], dtype=np.int32),                   # empty
        np.array([2], dtype=np.int64)                   # different dtype
    ])