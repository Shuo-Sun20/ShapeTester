import torch
from dataclasses import dataclass, field
from typing import Optional, List, Union

# 1. Valid test case
example_input = torch.randn(4, 2)
valid_test_case = {
    "inputs": [example_input],
    "out": None
}

# 2. Parameters affecting output shape (except "inputs")
# Only "out" parameter affects the output shape when provided

# 3. Value space analysis for "out" parameter
# Type: Optional[torch.Tensor]
# Value space: None (no output tensor provided) or torch.Tensor with various shapes
# Since it's continuous (tensor shapes), we discretize with boundary values and typical shapes

@dataclass
class InputSpace:
    """Contains parameters that affect output tensor shape with discretized value ranges"""
    
    # "out" parameter value space
    out: List[Optional[torch.Tensor]] = field(default_factory=lambda: [
        # Boundary/typical values for "out" parameter
        None,  # Default case - no output tensor
        torch.empty(4, 2),  # Same shape as input
        torch.empty(4, 2, dtype=torch.float64),  # Same shape, different dtype
        torch.empty(1, 2),  # Different shape - broadcastable (dim0=1)
        torch.empty(4, 1),  # Different shape - broadcastable (dim1=1)
        torch.empty(1, 1),  # Different shape - broadcastable (both dims=1)
        torch.empty(8, 4),  # Different shape - not broadcastable (for negative testing)
        torch.empty(2,),  # 1D tensor - not broadcastable (for negative testing)
    ])