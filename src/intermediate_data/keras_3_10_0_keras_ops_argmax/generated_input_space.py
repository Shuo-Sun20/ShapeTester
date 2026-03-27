import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Union

def call_func(inputs, axis=None, keepdims=False):
    x = inputs[0] if isinstance(inputs, list) else inputs
    return keras.ops.argmax(x=x, axis=axis, keepdims=keepdims)

test_tensor = keras.ops.convert_to_tensor(np.random.randn(3, 4, 5))
example_output = call_func(inputs=[test_tensor], axis=1, keepdims=False)

valid_test_case = {
    'inputs': [test_tensor],
    'axis': 1,
    'keepdims': False
}

@dataclass
class InputSpace:
    axis: List[Optional[int]] = field(default_factory=lambda: [None, 0, 1, -1, -2])
    keepdims: List[bool] = field(default_factory=lambda: [True, False])