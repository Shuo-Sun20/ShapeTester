import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List, Union, Optional

def call_func(height, width, inputs, data_format=None, name=None):
    layer_instance = keras.layers.CenterCrop(height=height, width=width, data_format=data_format, name=name)
    output_tensor = layer_instance(inputs)
    return output_tensor

# 1. Valid test case dictionary
valid_test_case = {
    'height': 224,
    'width': 224,
    'inputs': np.random.randn(2, 256, 256, 3).astype(np.float32),
    'data_format': 'channels_last',
    'name': None
}

# 2, 3 & 4. InputSpace dataclass with parameters affecting output shape
@dataclass
class InputSpace:
    # Height: integer target height, discretized to 5 values including boundaries
    height: List[int] = field(default_factory=lambda: [1, 64, 128, 224, 256])
    
    # Width: integer target width, discretized to 5 values including boundaries  
    width: List[int] = field(default_factory=lambda: [1, 64, 128, 224, 256])
    
    # Data format: discrete parameter with all possible values
    data_format: List[Optional[str]] = field(default_factory=lambda: ['channels_last', 'channels_first', None])