import tensorflow as tf
from dataclasses import dataclass, field
from typing import List

valid_test_case = {
    "inputs": tf.random.normal(shape=(3, 4))
}

@dataclass
class InputSpace:
    inputs: List[tf.Tensor] = field(default_factory=lambda: [
        tf.constant([1.0]),                              # 0D-like
        tf.constant([1.0, 2.0, 3.0]),                   # 1D
        tf.constant([[1.0, 2.0], [3.0, 4.0]]),          # 2D
        tf.constant([[[1.0, 2.0], [3.0, 4.0]]]),        # 3D
        tf.constant([[[[1.0, 2.0], [3.0, 4.0]]]])       # 4D
    ])