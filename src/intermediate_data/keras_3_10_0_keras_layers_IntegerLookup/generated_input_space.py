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
    "inputs": np.array([[12, 1138, 42], [42, 1000, 36]])
}

# Task 2: Parameters affecting output tensor shape (except "inputs")
# - output_mode: Determines if output matches input shape or becomes 2D
# - max_tokens: When pad_to_max_tokens=True, determines second dimension
# - pad_to_max_tokens: When True, forces output to (batch_size, max_tokens)
# - vocabulary: Its length determines second dimension when pad_to_max_tokens=False
# - num_oov_indices: Adds to vocabulary size
# - mask_token: Adds 1 to vocabulary size when output_mode="int"

# Task 3-4: Define InputSpace dataclass with discretized value ranges
@dataclass
class InputSpace:
    # Discrete parameters - list all possible values
    output_mode: List[str] = field(
        default_factory=lambda: ["int", "one_hot", "multi_hot", "count", "tf_idf"]
    )
    
    pad_to_max_tokens: List[bool] = field(
        default_factory=lambda: [True, False]
    )
    
    sparse: List[bool] = field(
        default_factory=lambda: [True, False]
    )
    
    invert: List[bool] = field(
        default_factory=lambda: [True, False]
    )
    
    vocabulary_dtype: List[str] = field(
        default_factory=lambda: ["int64"]  # Only supported dtype currently
    )
    
    # Discrete but with limited range
    num_oov_indices: List[int] = field(
        default_factory=lambda: [0, 1, 2, 3, 5]  # 0, boundary, and typical values
    )
    
    # Continuous parameters - discretized to 5+ values including boundaries
    max_tokens: List[Optional[int]] = field(
        default_factory=lambda: [None, 1, 10, 50, 100, 1000]  # Boundary: None and typical sizes
    )
    
    # Vocabulary length affects shape - represent via sample vocabularies
    vocabulary: List[List[int]] = field(
        default_factory=lambda: [
            [],  # Empty vocabulary
            [12],  # Single token
            [12, 36],  # Small vocabulary
            [12, 36, 1138, 42],  # Medium vocabulary (from example)
            list(range(10)),  # Sequential tokens
            list(range(100))  # Large vocabulary
        ]
    )
    
    # mask_token can be None or an integer
    mask_token: List[Optional[int]] = field(
        default_factory=lambda: [None, -1, 0, 1, 100]  # Common special tokens
    )
    
    # oov_token affects mapping but not shape in most cases
    oov_token: List[int] = field(
        default_factory=lambda: [-1, 0, 999999, -999]  # Common OOV tokens
    )
    
    # idf_weights only valid for tf_idf mode
    idf_weights: List[Optional[List[float]]] = field(
        default_factory=lambda: [
            None,
            [0.5],
            [0.25, 0.75, 0.6, 0.4],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [0.1] * 10,
            [0.01] * 100
        ]
    )