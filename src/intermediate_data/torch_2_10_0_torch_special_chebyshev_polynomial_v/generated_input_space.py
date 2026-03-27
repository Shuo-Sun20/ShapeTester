import torch
from dataclasses import dataclass, field
from typing import List, Union

# 1. Define valid_test_case
torch.manual_seed(0)
input_tensor = torch.randn(5)
n_tensor = torch.tensor(3)

valid_test_case = {
    'inputs': [input_tensor],
    'n': n_tensor,
    'out': None
}

# 3. Parameter analysis and value space construction
# Parameter 'n' affects output shape (through broadcasting with input)
# Possible types: int, torch.Tensor (scalar or multi-dimensional)

# 2 & 4. InputSpace dataclass definition
@dataclass
class InputSpace:
    # Parameter 'n' affects output shape
    n: List[Union[int, torch.Tensor]] = field(default_factory=lambda: [
        # Discrete values
        0,  # Boundary: minimum degree
        1,
        2,
        3,  # From valid_test_case
        4,
        5,
        10,  # Higher degree
        # Tensor types with different shapes/scalars
        torch.tensor(0),  # Scalar tensor
        torch.tensor(3),  # From valid_test_case
        torch.tensor(5),
        torch.tensor([0]),  # 1D tensor with single element
        torch.tensor([0, 1, 2]),  # 1D tensor
        torch.tensor([[0, 1], [2, 3]]),  # 2D tensor (for broadcasting)
        torch.tensor([], dtype=torch.int64),  # Empty tensor
    ])
    
    # Note: 'out' parameter doesn't affect shape, only 'n' does (besides 'inputs')