import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Union

# Task 1: Define valid_test_case
valid_test_case = {
    "max_tokens": None,
    "num_oov_indices": 1,
    "mask_token": None,
    "oov_token": -1,
    "vocabulary": [12, 36, 1138, 42],
    "vocabulary_dtype": "int64",
    "idf_weights": None,
    "invert": False,
    "output_mode": "int",
    "sparse": False,
    "pad_to_max_tokens": False,
    "name": None,
    "inputs": np.array([[1, 2, 3], [4, 5, 6]])
}

# Task 2 & 3 & 4: Define InputSpace class with parameters affecting output shape
@dataclass
class InputSpace:
    max_tokens: List[Optional[int]] = field(default_factory=lambda: [None, 10, 20, 50, 100])
    num_oov_indices: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    mask_token: List[Optional[int]] = field(default_factory=lambda: [None, 0, 100, 999, -999])
    output_mode: List[str] = field(default_factory=lambda: ["int", "one_hot", "multi_hot", "count", "tf_idf"])
    pad_to_max_tokens: List[bool] = field(default_factory=lambda: [True, False])
    vocabulary: List[List[int]] = field(default_factory=lambda: [
        [1, 2, 3],
        [10, 20, 30, 40, 50],
        [100, 200, 300, 400, 500, 600, 700],
        list(range(1000, 1020)),
        []
    ])