import keras
import numpy as np

def call_func(inputs, axis=None, keepdims=False):
    return keras.ops.logsumexp(inputs, axis=axis, keepdims=keepdims)

example_output = call_func(keras.ops.convert_to_tensor(np.random.normal(size=(4, 3))), axis=0, keepdims=True)