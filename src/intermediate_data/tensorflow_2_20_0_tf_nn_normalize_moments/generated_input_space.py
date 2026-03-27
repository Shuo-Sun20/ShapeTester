import tensorflow as tf
from dataclasses import dataclass, field
from typing import List, Union, Optional

valid_test_case = {
    'inputs': [tf.constant(100.0, dtype=tf.float32), 
               tf.random.normal(shape=(10,), dtype=tf.float32),
               tf.abs(tf.random.normal(shape=(10,), dtype=tf.float32))],
    'shift': None,
    'name': None
}

@dataclass
class InputSpace:
    shift: List[Union[None, tf.Tensor, float]] = field(default_factory=lambda: [
        None,
        0.0,
        1.0,
        tf.constant([0.5], dtype=tf.float32),
        tf.constant([1.5], dtype=tf.float32)
    ])