import keras
import numpy as np
from dataclasses import dataclass, field

def call_func(rate, inputs, training, noise_shape=None, seed=None):
    dropout_layer = keras.layers.Dropout(rate=rate, noise_shape=noise_shape, seed=seed)
    return dropout_layer(inputs, training=training)

# Task 1: Define valid_test_case
valid_test_case = {
    'rate': 0.5,
    'inputs': keras.ops.convert_to_tensor(np.random.rand(32, 10, 20)),
    'training': True,
    'noise_shape': None,
    'seed': None
}

# Task 2 & 3 & 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    noise_shape: list = field(default_factory=lambda: [
        None,
        (32, 1, 20),
        (32, 10, 1),
        (1, 10, 20),
        (32, 1, 1)
    ])