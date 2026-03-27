import keras
import numpy as np

def call_func(inputs):
    return keras.ops.isinf(inputs)

x = keras.ops.convert_to_tensor(np.random.randn(3, 4) * 10.0)
example_output = call_func(x)