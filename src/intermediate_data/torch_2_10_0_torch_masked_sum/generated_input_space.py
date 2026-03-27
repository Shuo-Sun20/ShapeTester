import torch
from dataclasses import dataclass, field
from typing import Union, Tuple, List, Optional

valid_test_case = {
    'inputs': [torch.tensor([[-3, -2, -1], [0, 1, 2]], dtype=torch.float32)],
    'dim': 1,
    'keepdim': False,
    'dtype': None,
    'mask': torch.tensor([[True, False, True], [False, False, False]])
}

@dataclass
class InputSpace:
    dim: List[Optional[Union[int, Tuple[int, ...]]]] = field(default_factory=lambda: [
        None, 0, 1, 2, -1, -2, -3, (0, 1), (0, 2), (1, 2), (0, 1, 2), 
        (0,), (1,), (2,), (-1, -2), (-1, -3), (-2, -3), (0, -1), (0, -2), (1, -2)
    ])
    keepdim: List[bool] = field(default_factory=lambda: [True, False])