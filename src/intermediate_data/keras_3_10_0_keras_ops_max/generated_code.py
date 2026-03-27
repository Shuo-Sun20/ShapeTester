import keras
import numpy as np

def call_func(inputs, axis=None, keepdims=False, initial=None):
    return keras.ops.max(inputs, axis=axis, keepdims=keepdims, initial=initial)

example_input = keras.random.uniform((3, 4, 5), minval=-5, maxval=5)
example_output = call_func(example_input, axis=1, keepdims=True)