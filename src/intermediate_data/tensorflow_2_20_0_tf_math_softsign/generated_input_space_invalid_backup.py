import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import List, Any

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": tf.random.normal(shape=(3, 4)),
    "name": None
}

# Task 2-4: Define InputSpace dataclass
@dataclass
class InputSpace:
    """Dataclass containing discretized value ranges for parameters 
       that affect the output tensor shape."""
    
    inputs: List[tf.Tensor] = field(default_factory=lambda: [
        tf.constant(0.0, dtype=tf.float32),  # scalar
        tf.ones(shape=(1,), dtype=tf.float32),  # 1D
        tf.ones(shape=(3, 2), dtype=tf.float32),  # 2D
        tf.ones(shape=(2, 3, 4), dtype=tf.float32),  # 3D
        tf.ones(shape=(1, 2, 3, 2), dtype=tf.float32)  # 4D
    ])
    
    # Only 'inputs' parameter affects output shape in tf.math.softsign