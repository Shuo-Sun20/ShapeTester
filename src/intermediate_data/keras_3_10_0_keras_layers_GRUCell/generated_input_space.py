import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

# Task 1: Valid test case
valid_test_case = {
    "units": 4,
    "activation": "tanh",
    "recurrent_activation": "sigmoid",
    "use_bias": True,
    "kernel_initializer": "glorot_uniform",
    "recurrent_initializer": "orthogonal",
    "bias_initializer": "zeros",
    "kernel_regularizer": None,
    "recurrent_regularizer": None,
    "bias_regularizer": None,
    "kernel_constraint": None,
    "recurrent_constraint": None,
    "bias_constraint": None,
    "dropout": 0.0,
    "recurrent_dropout": 0.0,
    "reset_after": True,
    "seed": None,
    "inputs": np.random.random((32, 10)),
    "states": np.random.random((32, 4)),
    "training": False
}

# Task 2 & 3: Identify shape-affecting parameters and their value spaces
# Only 'units' affects output shape (batch, units)

# Task 4: InputSpace dataclass
@dataclass
class InputSpace:
    units: List[int] = field(default_factory=lambda: [1, 4, 8, 16, 32])