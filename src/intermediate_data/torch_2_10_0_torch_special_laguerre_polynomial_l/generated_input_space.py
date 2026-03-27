import torch
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# 1. Valid test case
torch.manual_seed(42)
valid_test_case = {
    "inputs": [torch.randn(3, 4), torch.tensor([2])],
    "out": None
}

# 2. Parameters affecting output shape (besides "inputs")
# - out: Can accept pre-allocated tensor which must match output shape

# 3. Value space analysis
# a) out parameter:
#    - Type: Optional[torch.Tensor]
#    - Discrete values: None, pre-allocated tensors with various shapes/dtypes
#    - Value space includes:
#       1. None (default)
#       2. Tensor with same shape as expected output
#       3. Tensor with broadcastable shape
#       4. Tensor with different dtype but same shape
#       5. Tensor with same shape but different memory layout

# 4. InputSpace dataclass
@dataclass
class InputSpace:
    # out parameter value space
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,  # Default case
        # Same shape as expected output (3, 4) based on default inputs
        torch.zeros(3, 4),
        # Broadcastable shape cases
        torch.zeros(3, 4, dtype=torch.float64),
        torch.zeros(3, 4, dtype=torch.float32),
        # Different shape that can broadcast (1, 4) -> (3, 4)
        torch.zeros(1, 4),
        # Different shape that can broadcast (3, 1) -> (3, 4)
        torch.zeros(3, 1),
        # Same shape but different memory layout
        torch.zeros(3, 4).t().contiguous().t(),
        # Edge case: Empty tensor with compatible shape
        torch.zeros(0, 4),
        # Edge case: Scalar that can broadcast
        torch.tensor(0.0),
    ])