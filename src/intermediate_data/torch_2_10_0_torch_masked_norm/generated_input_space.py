import torch
from dataclasses import dataclass, field
from typing import Union, List, Tuple

valid_test_case = {
    'inputs': [torch.tensor([[-3., -2., -1.], [ 0., 1., 2.]])],
    'ord': 2.0,
    'dim': 1,
    'keepdim': False,
    'dtype': None,
    'mask': torch.tensor([[True, False, True], [False, False, False]])
}

@dataclass
class InputSpace:
    dim: List[Union[None, int, Tuple[int, ...]]] = field(default_factory=lambda: [
        None, 
        0, 
        1, 
        -1, 
        -2, 
        (0,), 
        (1,), 
        (0, 1), 
        (1, 0), 
        (0, -1), 
        (-1, 0), 
        (-2, -1), 
        (-1, -2),
        (-2,),
        (-1,)
    ])
    keepdim: List[bool] = field(default_factory=lambda: [False, True])