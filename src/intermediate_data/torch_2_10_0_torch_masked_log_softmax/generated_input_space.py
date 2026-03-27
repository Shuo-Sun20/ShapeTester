import torch
from dataclasses import dataclass

valid_test_case = {
    'inputs': torch.randn(2, 3),
    'dim': 1,
    'mask': torch.tensor([[True, False, True], [False, False, False]])
}

@dataclass
class InputSpace:
    # No parameters except 'inputs' affect the output tensor shape
    pass