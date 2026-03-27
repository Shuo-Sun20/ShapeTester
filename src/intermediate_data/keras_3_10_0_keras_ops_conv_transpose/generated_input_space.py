import keras
from dataclasses import dataclass, field
from typing import List, Optional, Union

# Task 1: Define valid_test_case
valid_test_case = {
    'inputs': keras.random.normal((2, 5, 5, 3)),
    'kernel': keras.random.normal((3, 3, 4, 3)),
    'strides': 2,
    'padding': 'same',
    'output_padding': None,
    'data_format': 'channels_last',
    'dilation_rate': 1
}

# Tasks 2-4: Define InputSpace dataclass with discretized parameter value spaces
@dataclass
class InputSpace:
    # Kernel affects output shape through its spatial dimensions and output_channels
    # Provide 5 kernel shapes with varying spatial sizes and output channels
    kernel: List[keras.KerasTensor] = field(default_factory=lambda: [
        keras.random.normal((1, 1, 4, 3)),  # Small spatial
        keras.random.normal((2, 2, 4, 3)),  # Small
        keras.random.normal((3, 3, 4, 3)),  # Medium
        keras.random.normal((4, 4, 4, 3)),  # Large
        keras.random.normal((5, 5, 4, 3))   # Large spatial
    ])
    
    # Strides: int or tuple, 5 discrete values
    strides: List[Union[int, tuple]] = field(default_factory=lambda: [
        1,  # No upsampling
        2,  # Double
        3,  # Triple
        (2, 1),  # Mixed
        (3, 2)   # Mixed larger
    ])
    
    # Padding: only 2 discrete values
    padding: List[str] = field(default_factory=lambda: [
        'valid',
        'same'
    ])
    
    # Output padding: int/tuple/None, 5 values including boundary
    output_padding: List[Optional[Union[int, tuple]]] = field(default_factory=lambda: [
        None,   # Default
        0,      # No extra padding
        1,      # Small padding
        2,      # Medium padding
        (1, 2)  # Mixed padding
    ])
    
    # Dilation rate: int or tuple, 5 discrete values
    dilation_rate: List[Union[int, tuple]] = field(default_factory=lambda: [
        1,      # No dilation
        2,      # Double dilation
        3,      # Triple dilation
        (2, 1), # Mixed dilation
        (3, 2)  # Mixed larger dilation
    ])