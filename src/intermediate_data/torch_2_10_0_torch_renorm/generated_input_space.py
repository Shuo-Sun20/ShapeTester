import torch
from dataclasses import dataclass, field
from typing import List, Optional, Union

# 1. valid_test_case definition
valid_test_case = {
    'inputs': [torch.randn(3, 4, 5)],
    'p': 2.0,
    'dim': 1,
    'maxnorm': 1.0,
    'out': None
}

# 2. Parameters affecting output shape: Only 'dim' affects execution slicing
# 3. Parameter value space construction:
#    - p: continuous, discretized to boundary + 5 typical values
#    - dim: discrete, limited by tensor dimensions
#    - maxnorm: continuous, discretized to boundary + 5 typical values
#    - out: discrete, either None or valid tensor

@dataclass
class InputSpace:
    """
    Value spaces for parameters affecting torch.renorm execution.
    Each field contains up to 5 discretized values.
    """
    p: List[float] = field(default_factory=lambda: [
        0.5, 1.0, 2.0, 3.0, float('inf')  # L0.5, L1, L2, L3, Linf norms
    ])
    
    dim: List[int] = field(default_factory=lambda: [
        -2, -1, 0, 1, 2  # Valid dimensions for typical 3D tensors
    ])
    
    maxnorm: List[float] = field(default_factory=lambda: [
        0.0, 0.5, 1.0, 5.0, 10.0  # Boundary (0) + typical values
    ])
    
    out: List[Optional[Union[torch.Tensor, str]]] = field(
        default_factory=lambda: [None, "tensor"]  # None or tensor placeholder
    )