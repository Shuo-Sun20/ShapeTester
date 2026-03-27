import torch
from dataclasses import dataclass, field
from typing import List, Optional

# Set seed for reproducibility
torch.manual_seed(42)

# Valid test case for call_func
valid_test_case = {
    "inputs": [torch.randn(3, 4), torch.randn(3, 4)],
    "out": None
}

@dataclass
class InputSpace:
    """Defines the value space for parameters affecting output shape in call_func."""
    
    # The 'out' parameter can affect output shape through broadcasting or if provided with a specific shape
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        None,  # No output tensor provided (default case)
        torch.empty((3, 4)),  # Same shape as broadcasted result
        torch.empty((1, 1)),  # Broadcastable to (3, 4)
        torch.empty((1, 4)),  # Broadcastable to (3, 4)
        torch.empty((3, 1)),  # Broadcastable to (3, 4)
        torch.empty((6, 8)),  # Different shape that will cause an error if used improperly
    ])