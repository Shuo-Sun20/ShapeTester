import torch
from dataclasses import dataclass
from typing import List, Optional

# Task 1: Define valid_test_case
valid_test_case = {
    'inputs': [torch.randn(2, 3)],  # Must be a list containing one tensor
    'dim': 1,
    'dtype': None,  # Can be None or a torch.dtype like torch.float64
    'mask': torch.tensor([[True, False, True], [False, False, False]])
}

# Tasks 2-4: Define InputSpace class with parameters affecting output shape
@dataclass
class InputSpace:
    # Parameters affecting output shape (excluding 'inputs'):
    # 1. dim - determines which dimension is operated on (doesn't change shape but affects computation)
    dim: List[int] = None
    # 2. dtype - doesn't affect shape
    dtype: List[Optional[torch.dtype]] = None
    # 3. mask - must be broadcastable to input shape, doesn't change output shape
    
    def __post_init__(self):
        # Initialize with discretized value ranges
        if self.dim is None:
            # For a 2x3 input tensor, valid dim values are [-2, -1, 0, 1]
            self.dim = [-2, -1, 0, 1]
        
        if self.dtype is None:
            # Discrete values: None (default), and common float dtypes
            self.dtype = [None, torch.float32, torch.float64, torch.float16]