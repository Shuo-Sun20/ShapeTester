import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Union, List, Tuple, Optional

# Task 1: Define valid_test_case
valid_test_case = {
    "inputs": np.random.rand(4, 10, 10, 128).astype(np.float32),
    "filters": 32,
    "kernel_size": 3,
    "strides": (1, 1),
    "padding": "valid",
    "data_format": None,
    "dilation_rate": (1, 1),
    "groups": 1,
    "activation": 'relu',
    "use_bias": True,
    "kernel_initializer": "glorot_uniform",
    "bias_initializer": "zeros",
    "kernel_regularizer": None,
    "bias_regularizer": None,
    "activity_regularizer": None,
    "kernel_constraint": None,
    "bias_constraint": None
}

# Task 2 & 3: Parameters affecting output shape and their value spaces
# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    """Dataclass containing parameters affecting Conv2D output shape with discretized value ranges"""
    
    # filters: int - affects output channel dimension (continuous, discretized)
    filters: List[int] = field(default_factory=lambda: [
        1,          # Minimum valid value
        16, 32, 64, # Typical values
        256,        # Larger typical value
        1024        # Boundary/large value
    ])
    
    # kernel_size: int or tuple/list of 2 ints - affects spatial dimensions
    kernel_size: List[Union[int, Tuple[int, int]]] = field(default_factory=lambda: [
        1,                    # Minimum size
        3, 5, 7,              # Typical sizes
        (1, 1), (3, 3),       # Tuple representations
        (5, 7), (7, 5),       # Non-square kernels
        (11, 11)              # Larger kernel
    ])
    
    # strides: int or tuple/list of 2 ints - affects spatial dimensions
    strides: List[Union[int, Tuple[int, int]]] = field(default_factory=lambda: [
        1,                    # Minimum stride
        2, 3, 4,              # Typical strides
        (1, 1), (2, 2),       # Tuple representations
        (1, 2), (2, 1),       # Non-square strides
        (3, 4), (4, 3)        # Larger non-square strides
    ])
    
    # padding: string - affects spatial dimensions
    padding: List[str] = field(default_factory=lambda: [
        "valid",              # No padding
        "same"                # Padding to maintain size
    ])
    
    # data_format: string - affects dimension ordering
    data_format: List[Optional[str]] = field(default_factory=lambda: [
        None,                 # Default (channels_last)
        "channels_last",      # Explicit channels_last
        "channels_first"      # channels_first
    ])
    
    # dilation_rate: int or tuple/list of 2 ints - affects spatial dimensions
    dilation_rate: List[Union[int, Tuple[int, int]]] = field(default_factory=lambda: [
        1,                    # No dilation
        2, 3, 4,              # Typical dilation rates
        (1, 1), (2, 2),       # Tuple representations
        (1, 2), (2, 1),       # Non-square dilations
        (3, 4)                # Larger non-square dilation
    ])
    
    # groups: int - affects filter grouping (though doesn't change output shape dimensions,
    # it's included as it affects how convolution is computed)
    groups: List[int] = field(default_factory=lambda: [
        1,                    # Standard convolution
        2, 4, 8,              # Typical group values
        16, 32                # Larger group values
    ])

# Example usage
if __name__ == "__main__":
    # Test that valid_test_case can be used
    def call_func(
        inputs,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        data_format=None,
        dilation_rate=(1, 1),
        groups=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None
    ):
        layer = keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint
        )
        return layer(inputs)
    
    # Test call with valid_test_case
    output = call_func(**valid_test_case)
    print(f"Output shape: {output.shape}")
    
    # Test InputSpace instantiation
    var = InputSpace()
    print(f"InputSpace parameters: {var}")