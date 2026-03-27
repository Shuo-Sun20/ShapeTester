import torch
from dataclasses import dataclass, field

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": [torch.zeros(3, 3), torch.arange(4).reshape(2, 2) + 1],
    "size": (2, 2),
    "stride": (1, 2),
    "storage_offset": 0
}

# Tasks 2-4: Define InputSpace dataclass
@dataclass
class InputSpace:
    # size: tuple of integers (dimensions) - determines the shape of the view/scatter region
    size: list = field(default_factory=lambda: [
        (1, 1), (2, 2), (3, 3), (1, 2), (2, 1)
    ])
    
    # stride: tuple of integers (step sizes) - determines the layout in memory
    stride: list = field(default_factory=lambda: [
        (1, 1), (1, 2), (2, 1), (2, 2), (3, 3)
    ])
    
    # storage_offset: integer - starting point in underlying storage
    storage_offset: list = field(default_factory=lambda: [
        0, 1, 2, 3, 4
    ])