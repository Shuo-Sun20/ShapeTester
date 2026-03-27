import torch
from dataclasses import dataclass

valid_test_case = {
    'inputs': torch.randn(2, 3, 3),
    'p': 2
}

@dataclass
class InputSpace:
    p: list = None
    
    def __post_init__(self):
        if self.p is None:
            self.p = [None, 'fro', 'nuc', float('inf'), float('-inf'), 1, -1, 2, -2]