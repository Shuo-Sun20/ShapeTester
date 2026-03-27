import torch
from dataclasses import dataclass, field

# Task 1: Define valid_test_case
valid_test_case = {
    'inputs': torch.randn(3, 4),  # 2D tensor
    'ord': 2,                      # 2-norm (largest singular value)
    'dim': (-2, -1),              # Last two dimensions (default)
    'keepdim': False,             # Do not keep reduced dimensions
    'dtype': torch.float32,       # Cast to float32
    'out': None                   # No output tensor provided
}

# Task 2 & 3: Identify shape-affecting parameters and their discretized value spaces
# Parameters affecting output shape: 'dim', 'keepdim'
# 'ord', 'dtype', and 'out' do not affect output shape

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    # 'dim': Tuple of two ints, can be positive or negative indices
    dim: list = field(default_factory=lambda: [
        # Boundary values for 2D matrices
        (-2, -1),    # Default: last two dimensions
        (0, 1),      # First two dimensions
        # Edge cases for negative indexing
        (-3, -2),    # Third-last and second-last dimensions (for >=3D tensors)
        (-1, -2),    # Reversed order of default
        # Positive indexing variations
        (1, 2),      # Second and third dimensions
        (0, -1),     # First and last dimensions
        # Special case: same dimension twice (should error, but included for completeness)
        (0, 0),
    ])
    
    # 'keepdim': Boolean flag
    keepdim: list = field(default_factory=lambda: [
        False,       # Default: reduce dimensions
        True,        # Keep reduced dimensions as size 1
    ])

# Example instantiation
var = InputSpace()