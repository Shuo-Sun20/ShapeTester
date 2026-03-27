import tensorflow as tf
from keras.layers import Lambda
from dataclasses import dataclass, field
from typing import Optional, Union, Callable, Tuple

def call_func(function, output_shape=None, mask=None, arguments=None, inputs=None):
    layer_instance = Lambda(function=function, output_shape=output_shape, mask=mask, arguments=arguments)
    return layer_instance(inputs)

# 1. Valid test case
valid_test_case = {
    "function": lambda x: x ** 2,
    "output_shape": None,
    "mask": None,
    "arguments": None,
    "inputs": tf.random.normal(shape=(2, 3))
}

# 2. Parameters affecting output shape: function, output_shape

# 3. Value spaces for parameters affecting output shape
@dataclass
class InputSpace:
    function: list[Callable] = field(default_factory=lambda: [
        # 5 functions with different output shapes
        lambda x: x[:, :2],          # (batch, 2)
        lambda x: x[..., tf.newaxis], # (batch, 3, 1)
        lambda x: tf.reduce_mean(x, axis=1, keepdims=True),  # (batch, 1)
        lambda x: tf.concat([x, x], axis=1),  # (batch, 6)
        lambda x: tf.reshape(x, (-1, 6))       # (batch, 6)
    ])
    
    output_shape: list[Optional[Union[Tuple, Callable]]] = field(default_factory=lambda: [
        None,
        (2,),
        (3, 1),
        lambda input_shape: (input_shape[0], input_shape[1] * 2),
        (6,)
    ])