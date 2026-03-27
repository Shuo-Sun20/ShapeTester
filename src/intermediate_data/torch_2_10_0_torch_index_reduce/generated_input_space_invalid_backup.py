import torch
from dataclasses import dataclass
from typing import List, Optional, Union

valid_test_case = {
    'inputs': torch.randn(5, 3),
    'dim': 0,
    'index': torch.tensor([0, 4, 2]),
    'source': torch.randn(3, 3),
    'reduce': 'mean',
    'include_self': True,
    'out': None
}

@dataclass
class InputSpace:
    dim: List[int] = None  # Dimension along which to index (affects shape compatibility)
    index: List[List[int]] = None  # List of index tensors of different valid shapes
    source: List[List[int]] = None  # List of source tensor shapes compatible with dim/index
    out: List[Optional[List[int]]] = None  # Optional output tensor shapes
    
    def __post_init__(self):
        if self.dim is None:
            # For 2D input tensor (5, 3), dim can be 0 or 1
            self.dim = [0, 1]
        
        if self.index is None:
            # Index must be 1D tensor with length matching source.shape[dim]
            # Provide 5 different index configurations
            self.index = [
                [0],           # Single element
                [0, 1],        # Two consecutive elements
                [0, 2],        # Two non-consecutive elements
                [0, 1, 2],     # Three consecutive elements
                [0, 1, 2, 3]   # Four consecutive elements
            ]
        
        if self.source is None:
            # Source tensor must match input shape except at dimension 'dim'
            # Provide 5 different source shapes compatible with example input (5, 3)
            self.source = [
                [1, 3],  # For dim=0, index length 1
                [2, 3],  # For dim=0, index length 2
                [3, 3],  # For dim=0, index length 3
                [5, 1],  # For dim=1, index length 1
                [5, 2]   # For dim=1, index length 2
            ]
        
        if self.out is None:
            # Output must have same shape as input (5, 3) if provided
            # Provide 5 different options including None
            self.out = [
                None,
                [5, 3],  # Same as input
                [5, 3],  # Same as input (different instance)
                [5, 3],  # Same as input (different instance)
                [5, 3]   # Same as input (different instance)
            ]