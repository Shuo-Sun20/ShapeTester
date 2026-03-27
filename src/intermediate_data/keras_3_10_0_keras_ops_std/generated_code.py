import keras
import numpy as np

def call_func(inputs, axis=None, keepdims=False):
    return keras.ops.std(inputs, axis=axis, keepdims=keepdims)

example_tensor = keras.ops.convert_to_tensor(np.random.randn(3, 4))
example_output = call_func(example_tensor, axis=1, keepdims=True)