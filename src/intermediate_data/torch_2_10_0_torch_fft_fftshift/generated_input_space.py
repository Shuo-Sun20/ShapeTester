import torch
from dataclasses import dataclass, field
from typing import Union, List, Tuple

valid_test_case = {
    'inputs': torch.randn(4, 4),
    'dim': (0, 1)
}

@dataclass
class InputSpace:
    dim: List[Union[int, Tuple[int, ...], None]] = field(default_factory=lambda: [
        None,
        0,
        1,
        -1,
        -2,
        (0,),
        (1,),
        (0, 1),
        (1, 0),
        (-1,),
        (-2,),
        (0, -1),
        (0, 1, -1),
        (-2, -1)
    ])