import torch
from dataclasses import dataclass, field
from typing import List

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": [torch.randn(1, 1, 12, 12)],
    "downscale_factor": 3
}

# Task 2 & 3 & 4: Define InputSpace dataclass with discretized parameter value ranges
@dataclass
class InputSpace:
    # downscale_factor: positive integer that must evenly divide input spatial dimensions
    downscale_factor: List[int] = field(
        default_factory=lambda: [
            1,  # Boundary value (no scaling)
            2,  # Common small factor
            3,  # Typical factor from example
            4,  # Medium factor
            6   # Larger factor for 12x12 input
        ]
    )