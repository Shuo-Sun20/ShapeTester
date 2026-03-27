import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, List

def call_func(inputs, axis=None, keepdims=False):
    return keras.ops.amax(x=inputs, axis=axis, keepdims=keepdims)

valid_test_case = {
    'inputs': keras.ops.convert_to_tensor(np.random.randn(3, 4)),
    'axis': 1,
    'keepdims': True
}

@dataclass
class InputSpace:
    axis: List[Optional[Union[int, Tuple[int, ...]]]] = field(
        default_factory=lambda: [None, 0, 1, -1, (0, 1)]
    )
    keepdims: List[bool] = field(
        default_factory=lambda: [True, False]
    )