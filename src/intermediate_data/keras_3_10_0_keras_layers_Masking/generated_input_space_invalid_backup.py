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

# Tasks 2-4: Only "inputs" affects output shape, "mask_value" only affects mask computation
@dataclass
class InputSpace:
    # No parameters other than inputs affect output shape
    # Return empty value ranges for mask_value since it doesn't affect shape
    mask_value: list[Any] = field(default_factory=list)