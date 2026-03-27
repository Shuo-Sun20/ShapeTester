import torch
from dataclasses import dataclass
from typing import Optional, List, Union

# 1. Define valid_test_case
torch.manual_seed(42)
valid_test_case = {
    'inputs': [torch.randn(3, 3)],
    'out': None
}

# 2. Parameters affecting output shape: Only 'out' (but indirectly, it must match the shape determined by 'inputs')
#    The output shape is determined by the input tensor A's batch dimensions (*, n, n) -> (*).
#    'out' doesn't change this shape, but must have compatible shape if provided.

# 3. Parameter value space analysis for 'out':
#    Type: Optional[torch.Tensor]
#    - None (default)
#    - Tensor with shape matching the output (batch dimensions of input A)
#      Since A's shape varies, we can't predefine exact tensors, but we can define strategies

# 4. Define InputSpace class
@dataclass
class InputSpace:
    # 'out' parameter discretization:
    # None (default) and example tensors for common batch dimensions
    # Note: Actual tensors would need to match input shape during testing
    out: List[Optional[torch.Tensor]] = None
    
    def __post_init__(self):
        if self.out is None:
            # Generate example 'out' tensors for typical batch scenarios
            # Includes None + tensors for different batch dimensions
            self.out = [
                None,  # Default
                # Single matrix (no batch) - scalar output
                torch.tensor(0.0, dtype=torch.float32),
                # Batch of 2 matrices
                torch.zeros(2, dtype=torch.float32),
                # Batch of 5 matrices  
                torch.zeros(5, dtype=torch.float32),
                # Larger batch
                torch.zeros(10, dtype=torch.float32),
                # Edge case: empty batch? (torch.linalg.det requires n>=1)
                torch.zeros(0, dtype=torch.float32),  # Represents no batch but valid shape
                # Different dtype to test compatibility
                torch.zeros(3, dtype=torch.double),
                # Already filled tensor
                torch.ones(3, dtype=torch.float32),
            ]

# The InputSpace can be instantiated with default values
# var = InputSpace()