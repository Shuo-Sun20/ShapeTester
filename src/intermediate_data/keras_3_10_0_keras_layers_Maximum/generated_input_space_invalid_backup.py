import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

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
    # Only parameter affecting output shape (excluding 'inputs')
    name: List[Optional[str]] = field(default_factory=lambda: [
        None,
        "maximum_layer",
        "layer1",
        "max_layer",
        ""
    ])