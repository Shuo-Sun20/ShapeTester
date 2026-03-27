import tensorflow as tf
from keras.layers import Lambda
from dataclasses import dataclass, field
from typing import Optional, Union, Callable, Tuple, List

def call_func(function, output_shape=None, mask=None, arguments=None, inputs=None):
    layer_instance = Lambda(function=function, output_shape=output_shape, mask=mask, arguments=arguments)
    return layer_instance(inputs)

# Define test functions for the function parameter
def identity(x):
    return x

def square(x):
    return x ** 2

def add_one(x):
    return x + 1.0

def sum_last_dim(x):
    return tf.reduce_sum(x, axis=-1, keepdims=True)

def expand_last_dim(x):
    return tf.expand_dims(x, axis=-1)

def flatten_except_batch(x):
    shape = tf.shape(x)
    return tf.reshape(x, [shape[0], -1])

# Define output_shape functions
def output_shape_flatten_except_batch(input_shape):
    if input_shape[0] is None:
        return (None, None)
    flattened = 1
    for dim in input_shape[1:]:
        flattened *= dim
    return (input_shape[0], flattened)

def output_shape_expand_last(input_shape):
    return input_shape + (1,)

def output_shape_same_as_input(input_shape):
    return input_shape

def output_shape_remove_last(input_shape):
    return input_shape[:-1]

def output_shape_double_last(input_shape):
    return input_shape[:-1] + (input_shape[-1] * 2,)

# 1. Valid test case
valid_test_case = {
    "function": square,
    "output_shape": None,
    "mask": None,
    "arguments": None,
    "inputs": tf.random.normal(shape=(2, 3))
}

# 2. Parameters affecting output shape: function, output_shape

# 3. Discretized value spaces

@dataclass
class InputSpace:
    function: List[Callable] = field(default_factory=lambda: [
        identity,
        square,
        add_one,
        sum_last_dim,
        expand_last_dim,
        flatten_except_batch
    ])
    
    output_shape: List[Optional[Union[Tuple, Callable]]] = field(default_factory=lambda: [
        None,
        (None, 4),
        (None, 3, 1),
        (None, 6),
        output_shape_flatten_except_batch,
        output_shape_expand_last,
        output_shape_same_as_input,
        output_shape_remove_last,
        output_shape_double_last
    ])