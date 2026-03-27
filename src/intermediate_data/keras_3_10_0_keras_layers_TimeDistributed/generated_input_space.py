import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List

# Define layers for value space
layer1 = keras.layers.Conv2D(64, (3, 3))
layer2 = keras.layers.Conv2D(64, (3, 3), padding='same')
layer3 = keras.layers.Conv2D(32, (5, 5), padding='valid')
layer4 = keras.layers.Flatten()
layer5 = keras.layers.MaxPooling2D((2, 2))
layer6 = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(128)
])

# Task 1: valid_test_case dictionary
valid_test_case = {
    "layer": layer1,
    "inputs": np.random.randn(32, 10, 128, 128, 3).astype(np.float32),
    "training": None,
    "mask": None,
    "name": None
}

# Task 4: InputSpace dataclass
@dataclass
class InputSpace:
    layer: List[keras.layers.Layer] = field(
        default_factory=lambda: [layer1, layer2, layer3, layer4, layer5, layer6]
    )