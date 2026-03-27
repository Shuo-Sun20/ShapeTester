import tensorflow as tf
from dataclasses import dataclass, field
from typing import Optional, List, Union, Tuple

valid_test_case = {
    'inputs': tf.random.normal(shape=[2, 3, 3, 4]),
    'diagonals_format': 'compact',
    'is_non_singular': None,
    'is_self_adjoint': None,
    'is_positive_definite': None,
    'is_square': None,
    'name': 'test_operator'
}

@dataclass
class InputSpace:
    diagonals_format: List[str] = field(
        default_factory=lambda: ['sequence', 'compact', 'matrix']
    )
    is_square: List[Optional[bool]] = field(
        default_factory=lambda: [None, True, False]
    )