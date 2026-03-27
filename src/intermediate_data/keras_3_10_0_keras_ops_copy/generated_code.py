import keras
import numpy as np

def call_func(inputs):
    return keras.ops.copy(inputs)

example_input = keras.random.normal(shape=(3, 4))
example_output = call_func(example_input)