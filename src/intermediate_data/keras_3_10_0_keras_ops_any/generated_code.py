import keras
import numpy as np

def call_func(inputs, axis=None, keepdims=False):
    return keras.ops.any(x=inputs, axis=axis, keepdims=keepdims)

x = keras.random.uniform(shape=(3, 4)) > 0.5
example_output = call_func(inputs=x, axis=1, keepdims=True)