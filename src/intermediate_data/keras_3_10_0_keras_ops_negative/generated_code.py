import keras
import numpy as np

def call_func(inputs):
    return keras.ops.negative(inputs)

x = keras.ops.convert_to_tensor(np.random.randn(2, 3))
example_output = call_func(x)