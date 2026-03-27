import torch
from dataclasses import dataclass, field
from typing import Optional, List

# Task 1: Define valid_test_case
torch.manual_seed(0)
valid_test_case = {
    'inputs': [torch.randn(3, 4) * 5],
    'out': None
}

# Tasks 2-4: Define InputSpace
@dataclass
class InputSpace:
    """Parameter space for torch.floor call via call_func"""
    
    # Only parameter affecting output shape (besides inputs) is 'out'
    # Discrete parameter: None or tensors with different properties
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,
        torch.zeros(3, 4),  # Same shape, zeros
        torch.ones(3, 4),   # Same shape, ones
        torch.empty(3, 4),  # Same shape, uninitialized
        torch.randn(3, 4)   # Same shape, random values
    ])