import torch
from dataclasses import dataclass, field
from typing import Optional, List, Union

# 1. Define valid_test_case
valid_test_case = {
    'inputs': [
        torch.tensor([4.0]),
        torch.tensor([3.0, 4.0, 5.0])
    ],
    'out': None
}

# 2. Parameters affecting output shape (except 'inputs')
# Only 'out' can affect output shape if provided, otherwise shape determined by broadcasting inputs

# 3. Discretized value space for parameters
# 'out' can be None or a tensor of compatible shape

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    out: List[Optional[Union[torch.Tensor, None]]] = field(default_factory=lambda: [
        None,
        torch.empty(()),  # Scalar tensor
        torch.empty(3),  # 1D tensor matching broadcast shape of example
        torch.empty(1, 3),  # 2D tensor
        torch.empty(2, 3),  # Different 2D tensor
        torch.empty(3, 1),  # 2D tensor transposed
        torch.tensor([0.0]),  # Scalar with value
        torch.tensor([0.0, 0.0, 0.0])  # 1D with values
    ])