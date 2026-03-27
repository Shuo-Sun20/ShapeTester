import torch
from dataclasses import dataclass, field
from typing import Union, List, Tuple

valid_test_case = {
    'padding': (1, 1, 2, 0),
    'inputs': torch.randn(2, 3, 5, 5)
}

@dataclass
class InputSpace:
    padding: List[Union[int, Tuple[int, int, int, int]]] = field(default_factory=lambda: [
        # Integer padding values
        -2, -1, 0, 1, 2, 3, 5, 10,
        
        # Tuple padding values (left, right, top, bottom)
        (0, 0, 0, 0),
        (1, 1, 1, 1),
        (2, 2, 2, 2),
        (0, 2, 0, 2),
        (2, 0, 2, 0),
        (0, 0, 1, 1),
        (1, 1, 0, 0),
        (1, 0, 1, 0),
        (0, 1, 0, 1),
        (1, 2, 3, 4),
        (4, 3, 2, 1),
        (-1, 0, 0, 0),
        (0, -1, 0, 0),
        (0, 0, -1, 0),
        (0, 0, 0, -1),
        (-1, -1, 0, 0),
        (0, 0, -1, -1),
        (-2, -2, 0, 0),
        (1, 1, 2, 0)  # From valid_test_case
    ])