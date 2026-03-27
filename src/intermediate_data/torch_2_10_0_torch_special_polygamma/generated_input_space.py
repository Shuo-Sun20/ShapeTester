import torch
from dataclasses import dataclass, field
from typing import Optional, List

# 1. Define valid_test_case
valid_test_case = {
    "n": 1,
    "inputs": [torch.randn(3, 4)],
    "out": None
}

# 2. Parameters affecting output shape (except 'inputs'): 
#    - 'out' (Optional[Tensor]) can affect output shape if provided
#    Note: 'n' (int) does NOT affect output shape as polygamma is element-wise

# 3. Value space analysis and discretization:
#    - 'out': Optional[Tensor]
#        * None (default)
#        * Tensor with same shape as input (output must match)
#        * Tensor with different shape (will cause runtime error)
#    Discretized to include both valid and invalid cases

# 4. InputSpace dataclass
@dataclass
class InputSpace:
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,  # Default case
        torch.randn(3, 4),  # Same shape as input
        torch.randn(2, 5),  # Different shape - will cause error
        torch.empty(3, 4),  # Same shape, uninitialized
        torch.zeros(3, 4),  # Same shape, zero-filled
        torch.ones(3, 4),   # Same shape, one-filled
    ])