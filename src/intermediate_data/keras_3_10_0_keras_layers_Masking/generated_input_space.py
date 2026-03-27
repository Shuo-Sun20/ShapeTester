import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Any

# Task 1: Define valid_test_case
samples, timesteps, features = 32, 10, 8
inputs = np.random.random([samples, timesteps, features]).astype(np.float32)
inputs[:, 3, :] = 0.
inputs[:, 5, :] = 0.
valid_test_case = {
    "mask_value": 0.0,
    "inputs": inputs
}

# Tasks 2-4: Create InputSpace dataclass
@dataclass
class InputSpace:
    mask_value: list = field(default_factory=lambda: [-100.0, -1.0, -0.5, 0.0, 0.5, 1.0, 100.0])