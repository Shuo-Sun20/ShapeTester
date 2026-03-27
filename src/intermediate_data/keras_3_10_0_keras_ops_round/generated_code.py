import keras
import numpy as np

def call_func(inputs, decimals=0):
    return keras.ops.round(inputs, decimals)

example_input = np.random.randn(3, 4).astype(np.float32)
example_output = call_func(example_input, decimals=2)