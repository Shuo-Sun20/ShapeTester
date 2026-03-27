import keras
import numpy as np

def call_func(inputs, axis=-1):
    return keras.ops.log_softmax(inputs, axis=axis)

x = np.random.randn(3, 4)
example_output = call_func([x])