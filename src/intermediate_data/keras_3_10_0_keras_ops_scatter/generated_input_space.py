import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

def call_func(inputs, shape):
    indices, values = inputs
    return keras.ops.scatter(indices, values, shape)

# 1. valid_test_case variable
np.random.seed(42)
indices = np.random.randint(0, 3, size=(5, 2))  # Converted to numpy array
values = np.random.random(size=5).astype(np.float32)
shape = (4, 4)

valid_test_case = {
    'inputs': [indices, values],
    'shape': shape
}

# 2-4. InputSpace dataclass
@dataclass
class InputSpace:
    # Parameter that affects output shape (excluding "inputs")
    shape: List[Tuple] = field(default_factory=lambda: [
        (1, 1),                 # Minimal 2D shape
        (2, 2),                 # Small shape
        (4, 4),                 # valid_test_case value
        (10, 10),               # Medium shape
        (100, 100),             # Large shape
        (1, 1, 1),              # 3D shape
        (2, 2, 2),              # 3D small
        (4, 4, 4),              # 3D medium
        (10, 10, 10),           # 3D large
        (2,),                   # 1D shape
        (100,),                 # 1D large
        (1, 10),                # Non-square 2D
        (10, 1),                # Non-square 2D
        (2, 3, 4),              # Non-cubic 3D
        (1, 1, 1, 1)            # 4D shape
    ])