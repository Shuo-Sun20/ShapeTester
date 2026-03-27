import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

def call_func(
    rate,
    noise_shape=None,
    seed=None,
    inputs=None,
    training=False
):
    layer = keras.layers.AlphaDropout(
        rate=rate,
        noise_shape=noise_shape,
        seed=seed
    )
    return layer(inputs, training=training)

# 1. Define valid_test_case
np.random.seed(42)
input_tensor = np.random.randn(2, 10, 8).astype(np.float32)

valid_test_case = {
    "rate": 0.2,
    "noise_shape": None,
    "seed": 42,
    "inputs": input_tensor,
    "training": True
}

# 2. & 3. Parameters affecting output shape (only noise_shape) with value spaces
@dataclass
class InputSpace:
    noise_shape: List[Optional[Tuple[Optional[int], ...]]] = field(
        default_factory=lambda: [
            None,
            (2, 1, 8),
            (2, 10, 1),
            (1, 10, 8),
            (2, 10, 8)
        ]
    )