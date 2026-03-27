import keras
import numpy as np
from typing import Optional, List, Union
from dataclasses import dataclass, field

valid_test_case = {
    "inputs": [
        keras.ops.convert_to_tensor(np.random.randn(2, 10, 9).astype(np.float32)),
        keras.ops.convert_to_tensor(np.random.randn(2, 10, 9).astype(np.float32))
    ],
    "sequence_length": 4,
    "sequence_stride": 2,
    "fft_length": 8,
    "length": None,
    "window": "hann",
    "center": True
}

@dataclass
class InputSpace:
    sequence_stride: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    fft_length: List[int] = field(default_factory=lambda: [4, 8, 16, 32, 64, 128, 256])
    length: List[Optional[int]] = field(default_factory=lambda: [None, 1, 10, 50, 100, 256, 512])
    center: List[bool] = field(default_factory=lambda: [True, False])