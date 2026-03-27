import keras
import numpy as np

def call_func(inputs):
    return keras.ops.hamming(inputs)

window_length = np.random.randint(1, 11)
inputs = keras.ops.convert_to_tensor(window_length)
example_output = call_func(inputs)