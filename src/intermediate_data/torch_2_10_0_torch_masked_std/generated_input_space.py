import torch
from dataclasses import dataclass, field
from typing import Union, Tuple, List, Optional

valid_test_case = {
    'inputs': torch.randn(2, 3),
    'dim': 1,
    'unbiased': True,
    'keepdim': False,
    'dtype': None,
    'mask': torch.tensor([[True, False, True], [False, True, False]])
}

@dataclass
class InputSpace:
    dim: List[Union[None, int, Tuple[int, ...]]] = field(default_factory=lambda: [
        None,
        0,
        1,
        -1,
        (0, 1),
        (1, 0),
        (0,),
        (-1, -2),
        (0, -2),
        (0, 1, 2)
    ])
    keepdim: List[bool] = field(default_factory=lambda: [True, False])