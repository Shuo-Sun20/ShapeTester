import tensorflow as tf
from dataclasses import dataclass, field
from typing import Union, List
import numpy as np

valid_test_case = {
    'inputs': tf.random.normal(shape=[7, 4], dtype=tf.float32),
    'segment_ids': tf.constant([0, 0, 1, 1, 2, 2, 2], dtype=tf.int32),
    'name': None
}

def call_func(inputs, segment_ids, name=None):
    return tf.math.segment_sum(data=inputs, segment_ids=segment_ids, name=name)

@dataclass
class InputSpace:
    segment_ids: List[Union[List[int], tf.Tensor]] = field(default_factory=lambda: [
        tf.constant([0, 0, 0], dtype=tf.int32),
        tf.constant([0, 0, 1, 1, 2], dtype=tf.int32),
        tf.constant([0, 1, 2, 3, 4], dtype=tf.int32),
        tf.constant([0, 0, 0, 0, 0], dtype=tf.int32),
        tf.constant([0, 0, 1, 2, 2], dtype=tf.int32)
    ])