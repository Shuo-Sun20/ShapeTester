import tensorflow as tf
from dataclasses import dataclass, field
from typing import Optional, Union, List, Tuple

valid_test_case = {
    'inputs': [tf.constant([1.0, 2.0, 3.0], dtype=tf.float32), 
               tf.constant([0.5, 1.0], dtype=tf.float32)],
    'eigvals_only': True,
    'select': 'a',
    'select_range': None,
    'tol': None,
    'name': None
}

@dataclass
class InputSpace:
    eigvals_only: List[bool] = field(default_factory=lambda: [True, False])
    select: List[str] = field(default_factory=lambda: ['a', 'v', 'i'])
    select_range: List[Optional[Union[Tuple[float, float], Tuple[int, int]]]] = field(
        default_factory=lambda: [None, (0.0, 1.0), (1.0, 2.0), (0, 0), (0, 1)]
    )