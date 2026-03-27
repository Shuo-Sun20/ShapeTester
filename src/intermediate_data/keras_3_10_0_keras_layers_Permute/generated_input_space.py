import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List

valid_test_case = {
    "dims": (2, 1),
    "inputs": keras.random.normal(shape=(5, 10, 64))
}

@dataclass
class InputSpace:
    dims: List[Tuple[int, ...]] = field(default_factory=lambda: [
        (1, 2),
        (2, 1),
        (1, 2, 3),
        (2, 1, 3),
        (1, 3, 2)
    ])