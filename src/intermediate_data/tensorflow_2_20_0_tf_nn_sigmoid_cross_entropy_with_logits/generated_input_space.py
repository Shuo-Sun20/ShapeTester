import tensorflow as tf
from dataclasses import dataclass, field
from typing import List, Any

valid_test_case = {
    'inputs': [
        tf.constant([[0.0, 0.5], [1.0, 0.7]]),  # labels
        tf.constant([[1.5, -0.5], [0.8, -1.2]]) # logits
    ],
    'name': None  # Optional parameter
}

@dataclass
class InputSpace:
    # The only parameter that affects output shape (except 'inputs' parameter)
    name: List[Any] = field(default_factory=lambda: [
        None,
        'custom_name_1',
        'custom_name_2',
        '',
        'test_sigmoid_cross_entropy'
    ])