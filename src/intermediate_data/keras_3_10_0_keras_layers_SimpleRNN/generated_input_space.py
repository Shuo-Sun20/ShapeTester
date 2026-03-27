import numpy as np
import keras
from dataclasses import dataclass, field
from typing import List, Optional, Union, Callable

# 1. Define valid_test_case with all call_func parameters
np.random.seed(42)
batch_size = 32
timesteps = 10
features = 8
inputs = np.random.random((batch_size, timesteps, features))

valid_test_case = {
    # Constructor parameters
    "units": 4,
    "activation": "tanh",
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
    "return_sequences": False,
    "return_state": False,
    "go_backwards": False,
    "stateful": False,
    "unroll": False,
    "seed": None,
    # Call parameters
    "inputs": inputs,
    "mask": None,
    "training": None,
    "initial_state": None
}

# 2. & 3. Parameters affecting output shape (excluding inputs) and their value spaces
@dataclass
class InputSpace:
    """Dataclass containing all parameters affecting SimpleRNN output shape."""
    
    # units: Positive integer, affects output dimension
    units: List[int] = field(default_factory=lambda: [
        1,           # Minimum boundary
        4,           # Typical small value
        16,          # Typical medium value
        64,          # Typical large value
        256,         # Large boundary value (practical upper limit)
    ])
    
    # return_sequences: Boolean, affects output timestep dimension
    return_sequences: List[bool] = field(default_factory=lambda: [
        True,
        False
    ])
    
    # return_state: Boolean, affects number of return values
    return_state: List[bool] = field(default_factory=lambda: [
        True,
        False
    ])