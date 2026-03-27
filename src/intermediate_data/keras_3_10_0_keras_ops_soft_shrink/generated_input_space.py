import numpy as np
import keras
from dataclasses import dataclass, field

def call_func(inputs, threshold=0.5):
    return keras.ops.soft_shrink(inputs, threshold)

np.random.seed(42)
x = np.random.randn(3, 4).astype(np.float32)

valid_test_case = {
    'inputs': x,
    'threshold': 0.5
}

@dataclass
class InputSpace:
    threshold: list = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.5, 1.0, 2.0])