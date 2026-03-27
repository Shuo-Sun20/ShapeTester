import tensorflow as tf
import keras
from dataclasses import dataclass, field
from typing import Optional, List, Union

# 1. Define valid_test_case
valid_test_case = {
    "inputs": tf.constant(["aaa", "bbb", "ccc", "ddd", "eee"]),
    "max_tokens": None,
    "num_oov_indices": 1,
    "mask_token": None,
    "oov_token": "[UNK]",
    "vocabulary": ['aaa', 'bbb', 'ccc', 'ddd', 'eee'],
    "idf_weights": None,
    "invert": False,
    "output_mode": "int",
    "pad_to_max_tokens": False,
    "sparse": False,
    "encoding": "utf-8",
    "name": None
}

# 2. & 3. Identify parameters affecting output shape and their value spaces
# Parameters: output_mode, pad_to_max_tokens, max_tokens, num_oov_indices

@dataclass
class InputSpace:
    """Dataclass containing all parameters affecting StringLookup output shape."""
    
    # output_mode: Discrete parameter
    output_mode: List[str] = field(
        default_factory=lambda: ["int", "one_hot", "multi_hot", "count", "tf_idf"]
    )
    
    # pad_to_max_tokens: Discrete parameter
    pad_to_max_tokens: List[bool] = field(
        default_factory=lambda: [True, False]
    )
    
    # max_tokens: Continuous parameter (discretized)
    max_tokens: List[Optional[int]] = field(
        default_factory=lambda: [None, 5, 10, 20, 50, 100]
    )
    
    # num_oov_indices: Discrete parameter
    num_oov_indices: List[int] = field(
        default_factory=lambda: [0, 1, 2, 3, 5, 10]
    )