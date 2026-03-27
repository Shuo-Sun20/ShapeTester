import keras
import numpy as np

def call_func(inputs, value_range=(0, 255)):
    layer = keras.layers.AutoContrast(value_range=value_range)
    return layer(inputs)

example_input = np.random.uniform(0, 255, size=(2, 32, 32, 3)).astype(np.float32)
example_output = call_func(example_input, value_range=(0, 255))