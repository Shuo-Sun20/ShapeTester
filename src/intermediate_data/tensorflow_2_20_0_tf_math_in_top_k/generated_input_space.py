import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import List

# 1. Define valid test case
valid_test_case = {
    "inputs": [
        tf.constant([0, 1, 3], dtype=tf.int32),
        tf.constant(
            [
                [1.2, -0.3, 2.8, 5.2],
                [0.1, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.3, 0.3]
            ],
            dtype=tf.float32
        )
    ],
    "k": 2
}

# 2. Parameters affecting output shape (besides inputs): k
# 3. Value spaces:
#    k: integer parameter, affects which predictions are considered
#       Discretized value space (boundary and typical values): [1, 2, 3, 4, 5]

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    k: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])