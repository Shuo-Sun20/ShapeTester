import numpy as np
import keras
from dataclasses import dataclass, field
from typing import List, Optional, Union, Callable

# Task 1: Define valid_test_case
batch_size = 32
features = 10
units = 4
example_inputs = np.random.random((batch_size, features))
example_states = np.random.random((batch_size, units))

valid_test_case = {
    "units": units,
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
    "kernel_constraint": None,
    "recurrent_constraint": None,
    "bias_constraint": None,
    "dropout": 0.0,
    "recurrent_dropout": 0.0,
    "seed": None,
    "inputs": example_inputs,
    "states": example_states,
    "training": False
}

# Task 4: Define InputSpace dataclass
@dataclass
class InputSpace:
    units: List[int] = field(default_factory=lambda: [1, 4, 16, 32, 64])