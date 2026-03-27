import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Union, List, Tuple

# 1. Define valid_test_case dictionary with proper input tensor
batch_size = 2
timesteps = 5
depth = 8
height = 8
width = 8
channels = 3
input_tensor = keras.random.normal((batch_size, timesteps, depth, height, width, channels))

valid_test_case = {
    'filters': 4,
    'kernel_size': 3,
    'strides': 1,
    'padding': "valid",
    'data_format': "channels_last",
    'dilation_rate': 1,
    'activation': "tanh",
    'recurrent_activation': "sigmoid",
    'use_bias': True,
    'kernel_initializer': "glorot_uniform",
    'recurrent_initializer': "orthogonal",
    'bias_initializer': "zeros",
    'unit_forget_bias': True,
    'kernel_regularizer': None,
    'recurrent_regularizer': None,
    'bias_regularizer': None,
    'activity_regularizer': None,
    'kernel_constraint': None,
    'recurrent_constraint': None,
    'bias_constraint': None,
    'dropout': 0.0,
    'recurrent_dropout': 0.0,
    'seed': None,
    'return_sequences': False,
    'return_state': False,
    'go_backwards': False,
    'stateful': False,
    'unroll': False,
    'inputs': input_tensor,
    'mask': None,
    'training': None,
    'initial_state': None
}

# 2. Define InputSpace dataclass with parameters that affect output shape
@dataclass
class InputSpace:
    # Parameters that affect output shape (excluding inputs)
    filters: List[int] = field(default_factory=lambda: [1, 4, 16, 32, 64])
    kernel_size: List[Union[int, Tuple[int, int, int]]] = field(
        default_factory=lambda: [1, (1,3,3), 3, (3,3,5), 5]
    )
    strides: List[Union[int, Tuple[int, int, int]]] = field(
        default_factory=lambda: [1, (1,1,2), 2, (1,2,2), (2,2,2)]
    )
    padding: List[str] = field(default_factory=lambda: ["valid", "same"])
    data_format: List[str] = field(default_factory=lambda: ["channels_last", "channels_first"])
    dilation_rate: List[Union[int, Tuple[int, int, int]]] = field(
        default_factory=lambda: [1, 2, (1,2,2), (2,2,2)]
    )
    return_sequences: List[bool] = field(default_factory=lambda: [True, False])
    return_state: List[bool] = field(default_factory=lambda: [True, False])