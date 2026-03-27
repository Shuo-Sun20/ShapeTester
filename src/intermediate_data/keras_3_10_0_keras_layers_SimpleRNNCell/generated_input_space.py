import numpy as np
from dataclasses import dataclass, field

# Task 1: Define valid_test_case
valid_test_case = {
    "units": 4,
    "activation": "tanh",
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
    "seed": None,
    "inputs": [np.random.random((32, 8)).astype(np.float32), 
               np.random.random((32, 4)).astype(np.float32)],
    "training": False
}

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    units: list = field(default_factory=lambda: [1, 2, 4, 8, 16])