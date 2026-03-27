import torch
from dataclasses import dataclass
from typing import Optional, Tuple, List

# 1. Define valid_test_case with working parameters
valid_test_case = {
    "inputs": torch.randn(10, 10, dtype=torch.complex64),
    "s": None,
    "dim": (-2, -1),
    "norm": None,
    "out": None
}

# 2. Parameters affecting output shape: 's' and 'dim'

# 3. & 4. Define InputSpace dataclass with discretized value spaces
@dataclass
class InputSpace:
    # Parameter 's' can be None or a tuple specifying output size
    s: List[Optional[Tuple[int, int]]] = None
    # Parameter 'dim' can be various valid dimension tuples
    dim: List[Tuple[int, int]] = None
    
    def __post_init__(self):
        if self.s is None:
            self.s = [
                None,  # Default case
                (8, 8),  # Smaller than input
                (10, 10),  # Same as input
                (12, 12),  # Larger than input (padding)
                (8, 12),  # Mixed dimensions
                (12, 8),  # Mixed dimensions reversed
                (-1, 10),  # No padding in first dimension
                (10, -1),  # No padding in second dimension
                (0, 0),  # Edge case: zero size
            ]
        
        if self.dim is None:
            self.dim = [
                (-2, -1),  # Default case
                (0, 1),  # Equivalent positive indices
                (1, 0),  # Reversed order
                (-1, -2),  # Reversed negative indices
                (0, -1),  # Mixed positive/negative
                (-2, 1),  # Mixed negative/positive
            ]