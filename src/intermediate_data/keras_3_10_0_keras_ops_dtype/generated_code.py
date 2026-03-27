import keras
import numpy as np

def call_func(inputs):
    return keras.ops.dtype(inputs)

example_input = keras.ops.convert_to_tensor(np.random.randn(8, 12))
example_output = keras.ops.convert_to_tensor(call_func(example_input))