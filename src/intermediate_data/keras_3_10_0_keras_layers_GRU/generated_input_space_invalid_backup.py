import keras
import numpy as np
from dataclasses import dataclass, field
from typing import Union, List, Optional

# 1. Define valid_test_case dictionary
valid_test_case = {
    "units": 4,
    "inputs": np.random.random((32, 10, 8)).astype(np.float32),
    "activation": "tanh",
    "recurrent_activation": "sigmoid",
    "use_bias": True,
    "kernel_initializer": "glorot_uniform",
    "recurrent_initializer": "orthogonal",
    "bias_initializer": "zeros",
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
    "reset_after": True,
    "use_cudnn": "auto",
    "mask": None,
    "training": None,
    "initial_state": None
}

# 2. Parameters affecting output tensor shape (except "inputs"):
# - units: Positive integer (dimensionality of output space)
# - return_sequences: Boolean (returns full sequence or last output)
# - return_state: Boolean (returns additional state tensors)

# 3. Value space analysis and discretization:
# units: Positive integer -> discrete values [1, 16, 32, 64, 128]
# return_sequences: Boolean -> [True, False]
# return_state: Boolean -> [True, False]

# 4. Define InputSpace dataclass
@dataclass
class InputSpace:
    units: List[int] = field(default_factory=lambda: [1, 16, 32, 64, 128])
    return_sequences: List[bool] = field(default_factory=lambda: [True, False])
    return_state: List[bool] = field(default_factory=lambda: [True, False])