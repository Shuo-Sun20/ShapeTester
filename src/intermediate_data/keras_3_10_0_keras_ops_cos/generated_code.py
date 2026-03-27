import keras
import numpy as np

def call_func(inputs):
    return keras.ops.cos(inputs)

example_tensor = keras.random.normal(shape=(3, 4))
example_output = call_func(example_tensor)