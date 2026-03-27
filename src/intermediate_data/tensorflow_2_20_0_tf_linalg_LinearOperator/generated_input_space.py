import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field

np.random.seed(42)
batch_size = 2
matrix = tf.constant(np.random.randn(batch_size, 3, 3).astype(np.float32))
x = tf.constant(np.random.randn(batch_size, 3, 4).astype(np.float32))

valid_test_case = {
    "inputs": [matrix, x],
    "is_non_singular": True,
    "is_self_adjoint": False,
    "is_positive_definite": None,
    "is_square": None,
    "name": "LinearOperatorTest",
    "adjoint": False,
    "adjoint_arg": False
}

@dataclass
class InputSpace:
    """Dataclass containing all parameters affecting output shape."""
    
    adjoint: list = field(default_factory=lambda: [False, True])
    adjoint_arg: list = field(default_factory=lambda: [False, True])