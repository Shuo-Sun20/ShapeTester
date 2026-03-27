import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List, Union

valid_test_case = {
    "inputs": np.random.randn(2, 10, 5).astype(np.float32),
    "sequence_lengths": np.random.randint(1, 11, size=(2,)).astype(np.int32),
    "strategy": "greedy",
    "beam_width": 100,
    "top_paths": 1,
    "merge_repeated": True,
    "mask_index": 0
}

@dataclass
class InputSpace:
    """All parameters that affect the shape of CTC decode output tensor"""
    
    strategy: List[str] = field(
        default_factory=lambda: ["greedy", "beam_search"]
    )
    
    top_paths: List[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 5, 10, 20, 50, 100]
    )