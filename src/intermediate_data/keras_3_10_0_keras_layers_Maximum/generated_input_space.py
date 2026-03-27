import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Union

def call_func(inputs, name=None):
    layer_instance = keras.layers.Maximum(name=name)
    return layer_instance(inputs)

# Task 1: valid_test_case dictionary
valid_test_case = {
    'inputs': [np.random.rand(2, 3, 4), np.random.rand(2, 3, 4)],
    'name': 'test_maximum'
}

# Task 2-4: InputSpace dataclass
@dataclass
class InputSpace:
    # The 'name' parameter is included because it's in call_func parameters
    # but it doesn't affect the output tensor shape
    name: List[Optional[str]] = field(
        default_factory=lambda: [
            None,
            'test_maximum',
            'maximum_layer',
            'max_layer',
            'custom_name_1',
            'custom_name_2'
        ]
    )