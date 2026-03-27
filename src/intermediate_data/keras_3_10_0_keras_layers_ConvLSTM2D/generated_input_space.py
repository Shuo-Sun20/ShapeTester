import keras
import numpy as np
from dataclasses import dataclass, field
from typing import List, Union, Tuple, Optional

# Definition of call_func (provided in the task)
def call_func(
    filters,
    kernel_size,
    inputs,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
    activation="tanh",
    recurrent_activation="sigmoid",
    use_bias=True,
    kernel_initializer="glorot_uniform",
    recurrent_initializer="orthogonal",
    bias_initializer="zeros",
    unit_forget_bias=True,
    kernel_regularizer=None,
    recurrent_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    recurrent_constraint=None,
    bias_constraint=None,
    dropout=0.0,
    recurrent_dropout=0.0,
    seed=None,
    return_sequences=False,
    return_state=False,
    go_backwards=False,
    stateful=False,
    unroll=False,
    mask=None,
    training=None,
    initial_state=None
):
    layer = keras.layers.ConvLSTM2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer,
        unit_forget_bias=unit_forget_bias,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        seed=seed,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll
    )
    return layer(inputs, mask=mask, training=training, initial_state=initial_state)

# 1. Define valid_test_case
batch_size = 2
timesteps = 5
rows = 32
cols = 32
channels = 3
input_tensor = np.random.randn(batch_size, timesteps, rows, cols, channels).astype(np.float32)

valid_test_case = {
    "filters": 32,
    "kernel_size": (3, 3),
    "inputs": input_tensor,
    "strides": 1,
    "padding": "valid",
    "data_format": None,
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
    "return_sequences": False,
    "return_state": False,
    "go_backwards": False,
    "stateful": False,
    "unroll": False,
    "mask": None,
    "training": None,
    "initial_state": None
}

# 2. Parameters affecting output shape (excluding inputs)
# Filters, kernel_size, strides, padding, data_format, dilation_rate, return_sequences, return_state

# 3. Define InputSpace with discretized value spaces
@dataclass
class InputSpace:
    filters: List[int] = field(default_factory=lambda: [1, 4, 16, 32, 64])
    kernel_size: List[Union[int, Tuple[int, int]]] = field(default_factory=lambda: [(1,1), (3,3), (5,5)])
    strides: List[Union[int, Tuple[int, int]]] = field(default_factory=lambda: [1, (2,2), (3,3)])
    padding: List[str] = field(default_factory=lambda: ["valid", "same"])
    data_format: List[Optional[str]] = field(default_factory=lambda: [None, "channels_last", "channels_first"])
    dilation_rate: List[Union[int, Tuple[int, int]]] = field(default_factory=lambda: [1, (2,2), (3,3)])
    return_sequences: List[bool] = field(default_factory=lambda: [False, True])
    return_state: List[bool] = field(default_factory=lambda: [False, True])