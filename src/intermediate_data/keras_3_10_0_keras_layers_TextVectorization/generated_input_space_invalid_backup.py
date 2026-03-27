import keras
import tensorflow as tf
from dataclasses import dataclass, field
from typing import List, Optional, Union, Tuple

# Task 1: Define valid test case
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

# Task 2-4: Define InputSpace dataclass
@dataclass
class InputSpace:
    """Dataclass containing parameters affecting output tensor shape with discretized value spaces"""
    
    # Discrete parameters (list all possible values)
    output_mode: List[str] = field(default_factory=lambda: ["int", "multi_hot", "count", "tf_idf"])
    pad_to_max_tokens: List[bool] = field(default_factory=lambda: [True, False])
    ragged: List[bool] = field(default_factory=lambda: [True, False])
    sparse: List[bool] = field(default_factory=lambda: [True, False])
    
    # Continuous parameters (discretized with 5 values including boundaries)
    max_tokens: List[Optional[int]] = field(default_factory=lambda: [None, 10, 50, 100, 500])
    output_sequence_length: List[Optional[int]] = field(default_factory=lambda: [None, 1, 4, 10, 32])
    
    # Parameters with mixed types (discretized)
    ngrams: List[Optional[Union[int, Tuple[int, ...]]]] = field(
        default_factory=lambda: [None, 1, 2, 3, (1, 2)]
    )
    split: List[Optional[str]] = field(
        default_factory=lambda: [None, "whitespace", "character"]
    )