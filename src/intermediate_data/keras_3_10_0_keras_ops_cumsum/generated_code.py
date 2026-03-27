import keras
import numpy as np

def call_func(inputs, axis=None, dtype=None):
    return keras.ops.cumsum(x=inputs, axis=axis, dtype=dtype)

x = keras.random.uniform(shape=(3, 4))
example_output = call_func(inputs=x, axis=0, dtype='float32')