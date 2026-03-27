import torch
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union

# Define valid test case
ccol_indices = torch.tensor([0, 2, 4], dtype=torch.int64)
row_indices = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
values = torch.tensor([1.0, 2.0, 3.0, 4.0])

valid_test_case = {
    'inputs': [ccol_indices, row_indices, values],
    'size': (2, 2),
    'dtype': torch.float64,
    'device': None,
    'pin_memory': False,
    'requires_grad': False,
    'check_invariants': None
}

@dataclass
class InputSpace:
    # Only 'size' parameter affects the output tensor shape (excluding 'inputs')
    # Parameter analysis and value space construction:
    # size: Can be None (inferred), or a sequence of ints (Tuple/List/torch.Size)
    # For this specific test case with given indices, we constrain sizes to be valid
    # (must have at least 2 dimensions, and dimensions must be ≥ max indices + 1)
    size: List[Optional[Tuple[int, ...]]] = field(default_factory=lambda: [
        None,                        # Infer from indices
        (2, 2),                      # Minimum valid size (from example)
        (3, 2),                      # Larger rows
        (2, 3),                      # Larger columns
        (3, 3),                      # Larger both dimensions
        (5, 5),                      # Much larger size
        (1, 2, 2),                   # With batch dimension (batch=1)
        (2, 2, 2),                   # Batch dimension > 1
        torch.Size([2, 2]),          # Using torch.Size object
    ])