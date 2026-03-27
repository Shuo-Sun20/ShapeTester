import keras
import numpy as np

def call_func(inputs, axis=None):
    return keras.ops.flip(x=inputs, axis=axis)

example_output = call_func(inputs=keras.ops.convert_to_tensor(np.random.randn(3, 4, 5)), axis=1)