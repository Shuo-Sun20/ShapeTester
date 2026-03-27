import numpy as np
import keras
from dataclasses import dataclass, field
from typing import List

# 1. Define valid_test_case dictionary
valid_test_case = {
    "units": 4,
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
    "return_sequences": False,
    "return_state": False,
    "go_backwards": False,
    "stateful": False,
    "unroll": False,
    "use_cudnn": "auto",
    "inputs": np.random.randn(32, 10, 8).astype(np.float32),
    "mask": None,
    "training": None,
    "initial_state": None
}

# 2 & 3. Parameters affecting output shape: units, return_sequences, return_state
# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    units: List[int] = field(default_factory=lambda: [1, 16, 32, 64, 128])
    return_sequences: List[bool] = field(default_factory=lambda: [True, False])
    return_state: List[bool] = field(default_factory=lambda: [True, False])