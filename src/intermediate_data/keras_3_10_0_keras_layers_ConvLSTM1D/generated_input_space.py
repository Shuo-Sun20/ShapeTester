import numpy as np
import keras
from dataclasses import dataclass, field
from typing import List

# Task 1: Define valid_test_case dict
batch_size = 4
timesteps = 10
rows = 32
channels = 3
inputs_tensor = np.random.randn(batch_size, timesteps, rows, channels).astype(np.float32)

valid_test_case = {
    "filters": 16,
    "kernel_size": 3,
    "strides": 1,
    "padding": "same",
    "data_format": "channels_last",
    "dilation_rate": 1,
    "activation": "tanh",
    "recurrent_activation": "sigmoid",
    "use_bias": True,
    "kernel_initializer": "glorot_uniform",
    "recurrent_initializer": "orthogonal",
    "bias_initializer": "zeros",
    "unit_forget_bias": True,
    "kernel_regularizer": None,
    "recurrent_regularizer": None,
    "bias_regularizer": None,
    "activity_regularizer": None,
    "kernel_constraint": None,
    "recurrent_constraint": None,
    "bias_constraint": None,
    "dropout": 0.0,
    "recurrent_dropout": 0.0,
    "seed": None,
    "return_sequences": True,
    "return_state": False,
    "go_backwards": False,
    "stateful": False,
    "unroll": False,
    "inputs": inputs_tensor,
    "initial_state": None,
    "mask": None,
    "training": False
}

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    filters: List[int] = field(default_factory=lambda: [1, 8, 16, 32, 64])
    kernel_size: List[int] = field(default_factory=lambda: [1, 3, 5, 7, 9])
    strides: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    padding: List[str] = field(default_factory=lambda: ["valid", "same"])
    data_format: List[str] = field(default_factory=lambda: ["channels_first", "channels_last"])
    dilation_rate: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    return_sequences: List[bool] = field(default_factory=lambda: [True, False])