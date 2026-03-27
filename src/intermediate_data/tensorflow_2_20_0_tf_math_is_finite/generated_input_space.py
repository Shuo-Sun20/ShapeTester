import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

valid_test_case = {
    "inputs": tf.constant([5.0, 4.8, 6.8, np.inf, np.nan], dtype=tf.float32),
    "name": None
}

@dataclass
class InputSpace:
    name: List[Optional[str]] = field(
        default_factory=lambda: [None, "test_op1", "test_op2", "test_op3", "test_op4", "test_op5"]
    )