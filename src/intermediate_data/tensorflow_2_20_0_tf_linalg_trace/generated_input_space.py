import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

valid_test_case = {
    "inputs": [tf.constant(np.random.randn(3, 4, 4), dtype=tf.float32)],
    "name": None
}

@dataclass
class InputSpace:
    name: Optional[List[Optional[str]]] = field(
        default_factory=lambda: [
            None,
            "trace_op_1",
            "trace_op_2",
            "my_trace",
            ""
        ]
    )