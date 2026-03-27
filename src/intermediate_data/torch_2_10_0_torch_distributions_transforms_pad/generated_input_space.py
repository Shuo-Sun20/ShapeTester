import torch
from dataclasses import dataclass, field
from typing import List, Tuple

valid_test_case = {
    'inputs': torch.randn(3, 3, 4, 2),
    'pad': (1, 1),
    'mode': 'constant',
    'value': 0.0
}

@dataclass
class InputSpace:
    pad: List[Tuple[int, ...]] = field(default_factory=lambda: [
        # Single dimension padding (tuple length 2)
        (0, 0),      # No padding
        (1, 1),      # Symmetric padding
        (0, 1),      # Asymmetric padding (right only)
        (2, 0),      # Asymmetric padding (left only)
        (3, 5),      # Large asymmetric padding
        
        # Two dimensions padding (tuple length 4)
        (0, 0, 0, 0),
        (1, 1, 1, 1),  # Symmetric 2D padding
        (1, 2, 3, 4),  # Asymmetric 2D padding
        (0, 5, 0, 3),  # Mixed zero and non-zero
        (10, 10, 10, 10),  # Large symmetric
        
        # Three dimensions padding (tuple length 6)
        (0, 0, 0, 0, 0, 0),
        (1, 1, 1, 1, 1, 1),  # Symmetric 3D padding
        (1, 2, 3, 4, 5, 6),  # Asymmetric 3D padding
        (0, 1, 0, 2, 0, 3),  # Alternating zeros
        (5, 5, 10, 10, 15, 15)  # Large symmetric with varying sizes
    ])