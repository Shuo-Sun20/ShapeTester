import tensorflow as tf
import keras
from dataclasses import dataclass, field
from typing import List, Optional, Union, Tuple, Any

# Task 1: Define a valid test case
valid_test_case = {
    "inputs": tf.constant([["hello world test example"],
                           ["keras tensorflow deep learning"],
                           ["machine learning ai model"]]),
    "max_tokens": 20,
    "standardize": "lower_and_strip_punctuation",
    "split": "whitespace",
    "ngrams": None,
    "output_mode": "int",
    "output_sequence_length": 4,
    "pad_to_max_tokens": False,
    "vocabulary": ["hello", "world", "test", "example", "keras", "tensorflow", 
                   "deep", "learning", "machine", "ai", "model"],
    "idf_weights": None,
    "sparse": False,
    "ragged": False,
    "encoding": "utf-8",
    "name": None
}

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    # Parameters that affect output shape
    max_tokens: List[Optional[int]] = field(default_factory=lambda: [None, 10, 20, 50, 100, 500])
    ngrams: List[Optional[Union[int, Tuple[int, ...]]]] = field(default_factory=lambda: [None, 1, 2, 3, (1,2), (2,3)])
    output_mode: List[str] = field(default_factory=lambda: ["int", "multi_hot", "count", "tf_idf"])
    output_sequence_length: List[Optional[int]] = field(default_factory=lambda: [None, 1, 2, 4, 8, 16, 32])
    pad_to_max_tokens: List[bool] = field(default_factory=lambda: [True, False])