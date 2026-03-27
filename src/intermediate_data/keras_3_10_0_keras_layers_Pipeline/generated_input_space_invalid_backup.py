import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List, Any, Optional, Union

# 1. Valid test case
valid_test_case = {
    "layers": [
        keras.layers.Rescaling(scale=1./255),
        keras.layers.RandomFlip(mode="horizontal"),
        keras.layers.RandomRotation(factor=0.1),
    ],
    "inputs": keras.random.uniform(shape=(4, 32, 32, 3), minval=0, maxval=1),
    "name": None,
    "training": None,
}

# 2. Parameters affecting output shape (except inputs):
# Only "layers" parameter affects output shape as different layers may change tensor dimensions

# 3. Parameter value space analysis:
# - layers: Discrete parameter with infinite possibilities. We'll create representative examples
#   covering different shape-changing scenarios
# - name: Does not affect shape (always None in this context)
# - training: Does not affect shape (affects random behavior but not dimensions)

# 4. InputSpace dataclass
@dataclass
class InputSpace:
    # The only parameter affecting output shape (except inputs) is "layers"
    layers: List[List[Any]] = field(default_factory=lambda: [
        # Case 1: No shape change
        [
            keras.layers.Rescaling(scale=1./255),
            keras.layers.RandomFlip(mode="horizontal"),
            keras.layers.RandomRotation(factor=0.1),
        ],
        # Case 2: Shape change via resizing
        [
            keras.layers.Rescaling(scale=1./255),
            keras.layers.RandomFlip(mode="horizontal"),
            keras.layers.Resizing(height=24, width=24),
        ],
        # Case 3: Shape change via cropping
        [
            keras.layers.RandomCrop(height=28, width=28),
            keras.layers.RandomFlip(mode="horizontal"),
        ],
        # Case 4: Multiple shape changes
        [
            keras.layers.RandomCrop(height=30, width=30),
            keras.layers.Resizing(height=20, width=20),
            keras.layers.RandomFlip(mode="horizontal"),
        ],
        # Case 5: Only preprocessing (no augmentation)
        [
            keras.layers.Rescaling(scale=1./255),
        ],
    ])