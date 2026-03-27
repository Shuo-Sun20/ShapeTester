import keras
import numpy as np

def call_func(dims, inputs):
    if isinstance(inputs, list):
        input_tensor = inputs[0]
    else:
        input_tensor = inputs
    permute_layer = keras.layers.Permute(dims)
    return permute_layer(input_tensor)

example_input = keras.random.normal(shape=(5, 10, 64))
example_output = call_func((2, 1), example_input)