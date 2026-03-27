import torch
from dataclasses import dataclass
from typing import Tuple, List, Optional

valid_test_case = {
    'inputs': [torch.randn(3, 3)],
    'size': (2, 2),
    'stride': (1, 2),
    'storage_offset': None
}

@dataclass
class InputSpace:
    size: List[Tuple[int, ...]] = None
    stride: List[Tuple[int, ...]] = None
    storage_offset: List[Optional[int]] = None
    
    def __post_init__(self):
        if self.size is None:
            self.size = [(1, 1), (2, 2), (3, 3), (1, 2), (2, 1)]
        if self.stride is None:
            self.stride = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1)]
        if self.storage_offset is None:
            self.storage_offset = [None, 0, 1, 2, 4]