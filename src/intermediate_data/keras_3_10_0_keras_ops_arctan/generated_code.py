import keras
import numpy as np

def call_func(inputs):
    return keras.ops.arctan(inputs)

input_tensor = keras.ops.convert_to_tensor(np.random.randn(3, 4))
example_output = call_func(input_tensor)