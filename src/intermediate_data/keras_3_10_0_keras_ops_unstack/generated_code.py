import keras
import numpy as np

def call_func(inputs, axis=0, num=None):
    return keras.ops.unstack(inputs, axis=axis, num=num)

x = keras.ops.array(np.random.randn(3, 4, 5))
example_output = call_func(x, axis=1)