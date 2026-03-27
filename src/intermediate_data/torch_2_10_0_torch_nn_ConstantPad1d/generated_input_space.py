import torch
from dataclasses import dataclass, field
from typing import Union, Tuple, List

# Task 1: Define valid_test_case
valid_test_case = {
    "padding": 2,
    "value": 3.5,
    "inputs": torch.randn(1, 2, 4)  # Shape (N, C, W_in) = (1, 2, 4)
}

# Task 3-4: Define InputSpace dataclass
@dataclass
class InputSpace:
    """
    Dataclass containing all parameters that affect output shape
    with their discretized value ranges.
    """
    # Parameter: padding (int or 2-tuple)
    # Values: boundary cases + typical values covering legal scenarios
    padding: List[Union[int, Tuple[int, int]]] = field(default_factory=lambda: [
        # Integer padding values (left=right)
        0,      # No padding boundary
        1,      # Minimal positive padding
        2,      # Small positive (from valid_test_case)
        5,      # Moderate positive
        10,     # Larger positive
        # Tuple padding values (left, right)
        (0, 0),   # No padding both sides
        (1, 0),   # Asymmetric - left only
        (0, 1),   # Asymmetric - right only
        (2, 2),   # Symmetric positive
        (1, 3),   # Asymmetric positive (example from docs)
        (5, 5),   # Larger symmetric
        (3, 7),   # Larger asymmetric
        (0, 10),  # Max right-only in typical range
        (10, 0),  # Max left-only in typical range
        # Boundary/edge cases for correctness
        -1,      # Negative padding (allowed, causes cropping)
        (-1, 0), # Mixed negative/zero
        (0, -1), # Mixed zero/negative
        (-2, 2), # Mixed negative/positive
        (2, -2), # Mixed positive/negative
    ])