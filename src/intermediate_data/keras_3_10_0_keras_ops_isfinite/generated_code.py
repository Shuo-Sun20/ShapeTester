import keras
import numpy as np

def call_func(inputs):
    return keras.ops.isfinite(inputs)

example_input = keras.random.normal((3, 4))
example_output = call_func(example_input)