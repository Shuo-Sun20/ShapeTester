import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union

def call_func(
    inputs,
    size,
    strides=None,
    dilation_rate=1,
    padding="valid",
    data_format=None
):
    return keras.ops.image.extract_patches(
        images=inputs,
        size=size,
        strides=strides,
        dilation_rate=dilation_rate,
        padding=padding,
        data_format=data_format
    )

example_input = np.random.random((2, 20, 20, 3)).astype("float32")
valid_test_case = {
    'inputs': example_input,
    'size': (5, 5),
    'strides': None,
    'dilation_rate': 1,
    'padding': 'valid',
    'data_format': None
}

# Task 4: InputSpace dataclass definition
@dataclass
class InputSpace:
    """
    Contains all parameters that affect the shape of extract_patches output,
    with discretized value spaces for each parameter.
    """
    size: List[Tuple[int, int]] = field(default_factory=lambda: [
        # Boundary values
        (1, 1),
        (1, 20),  # full width
        (20, 1),  # full height
        (20, 20),  # full image
        # Typical values from documentation
        (3, 3),
        (5, 5),  # From valid_test_case
        # Additional typical values
        (2, 2),
        (4, 4),
        (7, 7),
        (10, 10),
        # Non-square patches
        (3, 5),
        (5, 3),
        (7, 3),
        (3, 7),
        # Larger patches
        (15, 15)
    ])
    
    strides: List[Optional[Tuple[int, int]]] = field(default_factory=lambda: [
        # None (defaults to size)
        None,
        # Boundary values
        (1, 1),  # Minimum stride
        (20, 20),  # Full image stride (maximum for valid padding)
        (1, 20),  # Full width stride
        (20, 1),  # Full height stride
        # Typical values matching size
        (5, 5),  # From valid_test_case when strides=None
        (3, 3),
        # Additional typical values
        (2, 2),
        (4, 4),
        (7, 7),
        (10, 10),
        # Non-square strides
        (1, 2),
        (2, 1),
        (2, 3),
        (3, 2),
        (4, 2),
        (2, 4),
        # Stride larger than patch size (may result in fewer patches)
        (8, 8),
        (5, 10),
        (10, 5)
    ])
    
    dilation_rate: List[Union[int, Tuple[int, int]]] = field(default_factory=lambda: [
        # Boundary values
        1,  # Minimum/Default - From valid_test_case
        # Integer dilation rates
        2,
        3,
        4,
        5,
        # Larger dilation rates
        6,
        7,
        8,
        # Boundary cases for tuple
        (1, 1),
        # Non-square tuple dilation rates
        (1, 2),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 2),
        (3, 3),
        # Larger tuple dilation rates
        (4, 4),
        (1, 4),
        (4, 1)
    ])
    
    padding: List[str] = field(default_factory=lambda: [
        'valid',
        'same'  # Only two possible values based on documentation
    ])
    
    data_format: List[Optional[str]] = field(default_factory=lambda: [
        None,  # From valid_test_case, defaults to keras.config.image_data_format
        'channels_last',
        'channels_first'
    ])

# Verify InputSpace can be instantiated
var = InputSpace()