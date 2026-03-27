import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Union, Tuple, List

def call_func(inputs, pool_size, strides=None, padding="valid", data_format=None):
    return keras.ops.average_pool(inputs, pool_size, strides, padding, data_format)

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": keras.random.normal(shape=(2, 8, 8, 3)),
    "pool_size": (2, 2),
    "strides": (2, 2),
    "padding": "valid",
    "data_format": "channels_last"
}

# Task 3 & 4: Define InputSpace dataclass with discretized value spaces
@dataclass
class InputSpace:
    # pool_size: int or tuple/list of integers
    # For 2D input with spatial shape (8,8), pool_size must be <= spatial dimensions
    pool_size: List[Union[int, Tuple[int, ...]]] = field(default_factory=lambda: [
        1, 2, 3, 4, 5, 6, 7, 8,                    # int values (boundary: 1,8)
        (1, 1), (2, 2), (3, 3), (4, 4), (5, 5),    # square tuples
        (2, 1), (1, 2), (3, 2), (2, 3), (4, 2),    # rectangular tuples
        (8, 8), (1, 8), (8, 1)                     # boundary tuples
    ])
    
    # strides: None, int, or tuple/list of integers
    # When None, defaults to pool_size. Must be <= pool_size and >= 1
    strides: List[Union[None, int, Tuple[int, ...]]] = field(default_factory=lambda: [
        None,                                      # default case
        1, 2, 3, 4, 5, 6, 7, 8,                    # int values (boundary: 1,8)
        (1, 1), (2, 2), (3, 3), (4, 4), (5, 5),    # square tuples
        (1, 2), (2, 1), (2, 3), (3, 2), (4, 2),    # rectangular tuples
        (8, 8), (1, 8), (8, 1)                     # boundary tuples
    ])
    
    # padding: string with two discrete values
    padding: List[str] = field(default_factory=lambda: [
        "valid", "same"
    ])
    
    # data_format: string with two discrete values or None
    data_format: List[Union[None, str]] = field(default_factory=lambda: [
        None, "channels_last", "channels_first"
    ])

# Example instantiation
var = InputSpace()