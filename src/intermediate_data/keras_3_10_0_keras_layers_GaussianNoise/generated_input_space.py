import numpy as np
from dataclasses import dataclass, field
from typing import Any

# 1. Define valid_test_case variable
valid_test_case = {
    "stddev": 0.5,
    "inputs": np.random.randn(2, 10, 10, 3).astype('float32'),
    "training": True,
    "seed": 42
}

# 2 & 3 & 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # stddev: Continuous parameter affecting output values but not shape
    stddev: list[float] = field(default_factory=lambda: [0.0, 0.1, 0.5, 1.0, 2.0])
    # training: Boolean parameter that determines whether noise is applied
    training: list[bool] = field(default_factory=lambda: [True, False])
    # seed: Integer parameter affecting random number generation
    seed: list[Any] = field(default_factory=lambda: [None, 42, 100, 200, 300])