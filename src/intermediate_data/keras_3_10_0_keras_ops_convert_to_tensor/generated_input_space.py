import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

def call_func(inputs, dtype=None, sparse=None, ragged=None):
    return keras.ops.convert_to_tensor(x=inputs, dtype=dtype, sparse=sparse, ragged=ragged)

x = np.random.rand(3, 4, 5)
example_output = call_func(x, dtype='float32')

valid_test_case = {
    'inputs': x,
    'dtype': 'float32',
    'sparse': None,
    'ragged': None
}

@dataclass
class InputSpace:
    sparse: List[Optional[bool]] = field(default_factory=lambda: [None, True, False])
    ragged: List[Optional[bool]] = field(default_factory=lambda: [None, True, False])