import torch
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any

# Generate example input as in the provided snippet
T = torch.rand(10, 9, dtype=torch.float32)
t = torch.fft.ihfftn(T)

# Task 1: Define valid_test_case
valid_test_case = {
    'inputs': [t],
    's': T.size(),
    'dim': None,
    'norm': None,
    'out': None
}

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    # Parameter s: can be None or tuple of ints
    # Value space includes: None, default even output, odd/even lengths, 
    # boundary values, and dimension modifications
    s: List[Optional[Tuple[int, ...]]] = field(default_factory=lambda: [
        None,  # Default case
        (10, 8),  # Even output (default for input shape (10,5))
        (10, 9),  # Odd output (matches valid_test_case)
        (10, 1),  # Minimum odd length
        (10, 2),  # Minimum even length
        (10, 16),  # Power of 2 (typical for CUDA half precision)
        (10, 17),  # Odd non-power-of-2
        (5, 9),   # Different first dimension (smaller)
        (15, 9),  # Different first dimension (larger)
        (10, -1, 9),  # No padding in middle dimension (3D example)
    ])
    
    # Parameter dim: can be None or tuple of ints
    # Value space covers: None, all dimensions, subsets, and permutations
    # Note: Last dimension must be the half-Hermitian compressed dimension
    dim: List[Optional[Tuple[int, ...]]] = field(default_factory=lambda: [
        None,  # Transform all dimensions
        (0, 1),  # Explicit all dimensions (2D)
        (1, 0),  # Permuted dimensions (valid if last is half-Hermitian)
        (0,),    # Only transform first dimension (last dim remains as is)
        (-2, -1),  # Negative indexing
        (1,),    # Only transform last dimension (must be half-Hermitian)
        (0, 1, 2),  # 3D case (if input had 3 dimensions)
    ])

    # Parameter norm: does not affect output shape, but included for completeness
    # as it was identified as affecting output in the broader sense
    norm: List[Optional[str]] = field(default_factory=lambda: [
        None,      # Defaults to "backward"
        "backward",
        "forward",
        "ortho",
    ])