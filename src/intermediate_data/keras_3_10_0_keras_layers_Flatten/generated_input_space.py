import numpy as np
import keras
from dataclasses import dataclass, field
from typing import List, Optional, Union

# 1. Valid test case
x = np.random.random((2, 10, 64)).astype('float32')
valid_test_case = {
    'inputs': x,
    'data_format': None
}

# 2-4. InputSpace dataclass
@dataclass
class InputSpace:
    """Dataclass containing parameters that affect Flatten output shape"""
    
    # data_format is the only parameter besides 'inputs' that affects output shape
    data_format: List[Optional[str]] = field(
        default_factory=lambda: [None, 'channels_last', 'channels_first']
    )
    
    # Note: While 'inputs' shape affects output shape, it's excluded per instructions
    # The discretized value spaces for input shapes would need to be defined separately
    # if we were to include them

# Example usage
if __name__ == "__main__":
    # Test the valid test case
    def call_func(inputs, data_format=None):
        flatten_layer = keras.layers.Flatten(data_format=data_format)
        return flatten_layer(inputs)
    
    # Test with valid test case
    output = call_func(**valid_test_case)
    print(f"Output shape: {output.shape}")
    
    # Create InputSpace instance
    input_space = InputSpace()
    print(f"Data format options: {input_space.data_format}")