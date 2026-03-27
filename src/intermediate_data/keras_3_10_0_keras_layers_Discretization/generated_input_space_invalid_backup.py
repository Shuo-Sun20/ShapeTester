import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

# 1. Valid test case
valid_test_case = {
    "bin_boundaries": [-1.0, 0.0, 1.0],
    "num_bins": None,
    "epsilon": 0.01,
    "output_mode": "int",
    "sparse": False,
    "dtype": None,
    "name": None,
    "inputs": np.random.randn(5, 4)
}

# 2. Parameters affecting output shape (except inputs): output_mode, num_bins, bin_boundaries

# 3-4. InputSpace dataclass with discretized value ranges
@dataclass
class InputSpace:
    # Parameters affecting output tensor shape
    output_mode: List[str] = field(default_factory=lambda: ["int", "one_hot", "multi_hot", "count"])
    num_bins: List[Optional[int]] = field(default_factory=lambda: [2, 4, 8, 16, 32])
    bin_boundaries: List[Optional[List[float]]] = field(
        default_factory=lambda: [
            None,
            [0.0, 1.0],
            [-1.0, 0.0, 1.0],
            [-2.0, -1.0, 0.0, 1.0, 2.0],
            [-0.5, 0.0, 0.5, 1.0]
        ]
    )