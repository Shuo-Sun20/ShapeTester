import keras
import numpy as np

def call_func(inputs, axis=None):
    x1, x2 = inputs
    return keras.ops.append(x1, x2, axis=axis)

x1 = keras.ops.convert_to_tensor(np.random.randn(2, 3))
x2 = keras.ops.convert_to_tensor(np.random.randn(1, 3))
example_output = call_func([x1, x2], axis=0)