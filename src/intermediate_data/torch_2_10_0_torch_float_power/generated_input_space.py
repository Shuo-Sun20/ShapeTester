import torch
from dataclasses import dataclass, field
from typing import List, Optional

# Task 1: Define valid_test_case
torch.manual_seed(42)
valid_test_case = {
    "inputs": [torch.randn(3, 4), torch.randn(3, 4)],
    "out": None
}

# Task 2 & 3 & 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    """Dataclass containing all parameters that affect output tensor shape."""
    
    # 'out' parameter value space (discretized)
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,
        torch.zeros((3, 4), dtype=torch.float64),
        torch.ones((3, 4), dtype=torch.float64),
        torch.full((3, 4), 2.0, dtype=torch.float64),
        torch.full((3, 4), -1.0, dtype=torch.float64)
    ])