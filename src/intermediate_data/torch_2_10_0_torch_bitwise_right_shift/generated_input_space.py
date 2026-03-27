import torch
from dataclasses import dataclass, field
from typing import Optional, List
import random

# Set seeds for reproducibility
random.seed(42)
torch.manual_seed(42)

# 1. Define valid_test_case dictionary
input_tensor = torch.randint(-128, 127, (3,), dtype=torch.int8)
other_tensor = torch.randint(0, 7, (3,), dtype=torch.int8)
valid_test_case = {
    'inputs': [input_tensor, other_tensor],
    'out': None
}

# 2. & 3. Parameters affecting output shape (excluding 'inputs'):
# Only 'out' parameter affects output shape.

@dataclass
class InputSpace:
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,
        torch.tensor([0, 0, 0], dtype=torch.int8),
        torch.tensor([0, 0, 0], dtype=torch.int16),
        torch.tensor([0, 0, 0], dtype=torch.int32),
        torch.tensor([0, 0, 0], dtype=torch.int64)
    ])