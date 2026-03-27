import keras
import numpy as np

def call_func(inputs):
    return keras.ops.arcsinh(inputs)

x = keras.ops.convert_to_tensor(np.random.randn(3, 4))
example_output = call_func(x)