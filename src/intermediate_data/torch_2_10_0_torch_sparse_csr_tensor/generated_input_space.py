import torch
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple

# Task 1: Define valid_test_case
crow_indices = torch.tensor([0, 2, 4], dtype=torch.int64)
col_indices = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
values = torch.randn(4)

valid_test_case = {
    'inputs': [crow_indices, col_indices, values],
    'size': (2, 2),
    'dtype': None,
    'device': None,
    'pin_memory': False,
    'requires_grad': False,
    'check_invariants': None
}

# Task 2 & 3: Parameter analysis and value space construction
# Parameters affecting output tensor shape (excluding "inputs"):
# 1. size: Directly determines the output shape
# 2. dtype: Can affect shape through type promotion (but not spatial dimensions)
# 3. device/pin_memory/requires_grad/check_invariants don't affect shape

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    # Parameter 1: size - affects spatial dimensions
    # Discretized value space covering boundary cases and typical scenarios
    size: Optional[Union[Tuple[int, int], Tuple[int, int, int]]] = field(
        default_factory=lambda: [
            None,                        # Infer size from indices
            (2, 2),                      # Minimal non-trivial (from valid_test_case)
            (5, 3),                      # Small rectangular
            (100, 100),                  # Medium square
            (1, 1000),                   # Wide matrix
            (1000, 1),                   # Tall matrix
            (0, 0),                      # Empty matrix (boundary)
            (2, 3, 4),                   # 3D batch example
            (1, 5, 5),                   # Batch size 1
            (10, 0, 10),                 # Zero rows (boundary)
            (10, 10, 0),                 # Zero columns (boundary)
        ]
    )
    
    # Parameter 2: dtype - can affect shape through type promotion
    # Discrete value space covering all relevant torch dtypes
    dtype: Optional[torch.dtype] = field(
        default_factory=lambda: [
            None,                        # Infer from values
            torch.float32,               # Default floating
            torch.float64,               # Double precision
            torch.complex64,             # Complex single
            torch.complex128,            # Complex double
            torch.int8,                  # Integer types
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.bool,                  # Boolean
            torch.half,                  # Half precision
            torch.bfloat16,              # Brain floating
        ]
    )
    
    # Note: device, pin_memory, requires_grad, check_invariants don't affect shape
    # but are included for completeness in the parameter list
    device: Optional[torch.device] = field(
        default_factory=lambda: [
            None,                        # Current default device
            torch.device('cpu'),         # CPU (always available)
        ]
    )
    
    pin_memory: bool = field(
        default_factory=lambda: [False, True]
    )
    
    requires_grad: bool = field(
        default_factory=lambda: [False, True]
    )
    
    check_invariants: Optional[bool] = field(
        default_factory=lambda: [None, False, True]
    )

# Example instantiation
var = InputSpace()