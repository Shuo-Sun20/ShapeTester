import numpy as np
import keras
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

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

# 2 & 3 & 4. Define InputSpace dataclass with parameters affecting output shape
@dataclass
class InputSpace:
    """
    Dataclass containing all parameters that affect the output shape of LSTM layer.
    Each field contains a list of possible discrete values or discretized values.
    """
    
    # units: Positive integer, dimensionality of output space
    # Discretized values covering typical range and boundary
    units: List[int] = field(default_factory=lambda: [1, 4, 8, 16, 32, 64, 128])
    
    # return_sequences: Boolean, affects output shape dimension
    # Discrete: all possible values
    return_sequences: List[bool] = field(default_factory=lambda: [True, False])
    
    # return_state: Boolean, affects number of return values
    # Discrete: all possible values
    return_state: List[bool] = field(default_factory=lambda: [True, False])
    
    # go_backwards: Boolean, affects sequence processing direction
    # Discrete: all possible values
    go_backwards: List[bool] = field(default_factory=lambda: [True, False])
    
    # stateful: Boolean, affects state management between batches
    # Discrete: all possible values
    stateful: List[bool] = field(default_factory=lambda: [True, False])
    
    # unroll: Boolean, affects computation graph
    # Discrete: all possible values
    unroll: List[bool] = field(default_factory=lambda: [True, False])

# Example instantiation
var = InputSpace()