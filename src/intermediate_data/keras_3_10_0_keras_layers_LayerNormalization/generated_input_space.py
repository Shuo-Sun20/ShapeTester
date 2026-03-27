import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Union

def call_func(
    inputs,
    axis=-1,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer="zeros",
    gamma_initializer="ones",
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None
):
    layer = keras.layers.LayerNormalization(
        axis=axis,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer=beta_initializer,
        gamma_initializer=gamma_initializer,
        beta_regularizer=beta_regularizer,
        gamma_regularizer=gamma_regularizer,
        beta_constraint=beta_constraint,
        gamma_constraint=gamma_constraint
    )
    return layer(inputs)

# 1. Valid test case
valid_test_case = {
    "inputs": np.random.randn(2, 5, 10, 8).astype(np.float32),
    "axis": [2, 3],
    "epsilon": 0.001,
    "center": True,
    "scale": True,
    "beta_initializer": "zeros",
    "gamma_initializer": "ones",
    "beta_regularizer": None,
    "gamma_regularizer": None,
    "beta_constraint": None,
    "gamma_constraint": None
}

# 2. Parameters affecting output shape: axis
# 3. Value space definition for axis (maximum 5 values)
@dataclass
class InputSpace:
    # Axis can be integer or list/tuple of integers
    # For 4D input (batch, height, width, channel), common valid values:
    axis: List[Union[int, Tuple[int, ...]]] = field(
        default_factory=lambda: [
            -1,  # Last dimension (channels)
            1,   # Height dimension
            [1, 2, 3],  # All spatial dimensions
            [2, 3],  # Width and channels
            [1, 2]   # Height and width
        ]
    )