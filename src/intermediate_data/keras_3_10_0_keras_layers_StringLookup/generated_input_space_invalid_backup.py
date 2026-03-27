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

@dataclass
class InputSpace:
    """Dataclass containing all parameters that affect output tensor shape."""
    
    max_tokens: List[Optional[int]] = field(
        default_factory=lambda: [None, 10, 20, 30, 40]
    )
    
    num_oov_indices: List[int] = field(
        default_factory=lambda: [0, 1, 2, 3, 4]
    )
    
    output_mode: List[str] = field(
        default_factory=lambda: ["int", "one_hot", "multi_hot", "count", "tf_idf"]
    )
    
    pad_to_max_tokens: List[bool] = field(
        default_factory=lambda: [True, False]
    )
    
    vocabulary: List[Optional[List[str]]] = field(
        default_factory=lambda: [
            None,
            ['a', 'b'],
            ['a', 'b', 'c', 'd', 'e'],
            ['a', 'b', 'c', 'd', 'e', 'f', 'g'],
            ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
        ]
    )