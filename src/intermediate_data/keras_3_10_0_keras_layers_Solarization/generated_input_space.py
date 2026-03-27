import numpy as np
import keras
from dataclasses import dataclass, field
from typing import Union, Tuple, List, Optional

# 1. Define valid_test_case
batch_size = 4
height = 32
width = 32
channels = 3
random_images = np.random.randint(
    low=0,
    high=256,
    size=(batch_size, height, width, channels),
    dtype=np.uint8,
)

valid_test_case = {
    'inputs': random_images,
    'addition_factor': 0.25,
    'threshold_factor': 0.5,
    'value_range': (0, 255),
    'seed': 42
}

# 4. Define InputSpace class
@dataclass
class InputSpace:
    addition_factor: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.25, 0.5, 1.0])
    threshold_factor: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.25, 0.5, 1.0])