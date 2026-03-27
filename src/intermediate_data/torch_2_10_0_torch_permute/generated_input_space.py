import torch
from dataclasses import dataclass, field
from typing import Tuple

# 1. Define valid_test_case
valid_test_case = {
    "inputs": [torch.randn(2, 3, 5)],
    "dims": (2, 0, 1)
}

# 2-4. Define InputSpace dataclass
@dataclass
class InputSpace:
    """
    All parameters that affect the shape of the output tensor (except 'inputs').
    For torch.permute, only 'dims' determines output shape.
    Discretized to 5 representative permutations for 3D tensors.
    """
    dims: list[Tuple[int, ...]] = field(
        default_factory=lambda: [
            (0, 1, 2),  # Original order
            (0, 2, 1),  # Swap last two
            (1, 0, 2),  # Swap first two
            (2, 0, 1),  # Example from documentation
            (1, 2, 0),  # Rotate dimensions
        ]
    )