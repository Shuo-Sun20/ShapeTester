import torch
from dataclasses import dataclass, field
from typing import Union, List, Tuple, Optional

valid_test_case = {
    "inputs": [torch.fft.rfft2(torch.randn(10, 9))],  # Input tensor after rfft2
    "s": (10, 9),                            # Original signal shape
    "dim": (-2, -1),                         # Transform dimensions
    "norm": "backward",                      # Normalization mode
    "out": None                              # Output tensor
}

@dataclass
class InputSpace:
    # Parameters that affect output shape
    s: List[Optional[Tuple[int, ...]]] = field(default_factory=lambda: [
        None,
        (8, 8),      # Even-even, smaller
        (10, 9),     # Even-odd, original
        (10, 10),    # Even-even, larger
        (9, 9),      # Odd-odd
        (10, 8),     # Even-even, different
        (12, 12),    # Larger even
        (7, 7),      # Smaller odd
        (10, 15),    # Different sizes
        (20, 20),    # Larger square
        (5, 5),      # Smaller square
        (10, -1),    # Partial specification
        (-1, 9),     # Partial specification
        (12, -1),    # Partial with larger first
        (-1, 15)     # Partial with larger second
    ])
    
    dim: List[Tuple[int, ...]] = field(default_factory=lambda: [
        (-2, -1),     # Default
        (0, 1),       # Same dimensions, positive indices
        (1, 0),       # Reversed order
        (-1, -2),     # Reversed negative indices
        (0, -1),      # Mixed positive/negative
        (-2, 1),      # Mixed negative/positive
        (2, 3),       # Higher dimensions for 4D input
        (-3, -2),     # Negative indices for 4D input
        (1, 2),       # Middle dimensions
        (-4, -3)      # First two dimensions for 4D
    ])