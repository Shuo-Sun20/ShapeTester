import keras
from dataclasses import dataclass, field
from typing import List, Optional, Union
import numpy as np

valid_test_case = {
    "inputs": keras.random.normal(shape=(10,)),
    "fft_length": None
}

@dataclass
class InputSpace:
    # Only parameter that affects output shape (besides inputs parameter)
    fft_length: List[Optional[int]] = field(
        default_factory=lambda: [
            None,  # Default: inferred from input length
            1,     # Minimum: fft_length = 1 (boundary)
            2,     # Smallest that produces complex output (boundary)
            5,     # Less than input length (typical)
            10,    # Equal to input length (typical, same as None)
            15,    # Greater than input length (typical)
            20,    # Much greater than input length (boundary)
            0,     # Edge case: fft_length = 0 (crops everything)
            8,     # Even number less than input (typical)
            9,     # Odd number less than input (typical)
        ]
    )