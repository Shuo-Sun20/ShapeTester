import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Union

def call_func(inputs, num_bins, mask_value=None, salt=None, output_mode='int', sparse=False):
    layer = keras.layers.Hashing(num_bins=num_bins, mask_value=mask_value, salt=salt, output_mode=output_mode, sparse=sparse)
    return layer(inputs)

valid_test_case = {
    'inputs': np.array([['ABC'], ['DEF'], ['GHI'], ['JKL'], ['MNO']], dtype=object),
    'num_bins': 10,
    'mask_value': None,
    'salt': None,
    'output_mode': 'int',
    'sparse': False
}

@dataclass
class InputSpace:
    num_bins: List[int] = field(default_factory=lambda: [1, 2, 5, 10, 100, 1000])
    output_mode: List[str] = field(default_factory=lambda: ["int", "one_hot", "multi_hot", "count"])
    sparse: List[bool] = field(default_factory=lambda: [False, True])