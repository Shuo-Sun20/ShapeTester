import torch
import random
from dataclasses import dataclass, field
from typing import Optional, List, Union

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)

# Create random input tensors
input_tensor = torch.randn(3, 4, dtype=torch.float32) * 10
other_tensor = torch.randn(4, dtype=torch.float32) * 5 + 1

# 1. Define valid_test_case
valid_test_case = {
    'inputs': [input_tensor, other_tensor],
    'rounding_mode': None,
    'out': None
}

# 2. Identify parameters affecting output shape (except "inputs")
# The parameters are: 'out'

# 3. Value space construction
# rounding_mode: discrete parameter
# out: can be None or a tensor with compatible shape/dtype

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    # Parameters that can affect output shape
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,
        torch.zeros(3, 4, dtype=torch.float32),      # Same shape as broadcasted result
        torch.ones(3, 4, dtype=torch.float32),       # Same shape, different values
        torch.empty(3, 4, dtype=torch.float32),      # Uninitialized
        torch.full((3, 4), 5.0, dtype=torch.float32) # Filled tensor
    ])