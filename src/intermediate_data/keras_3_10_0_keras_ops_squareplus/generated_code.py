import keras
import numpy as np

def call_func(inputs, b=4):
    return keras.ops.squareplus(inputs, b)

example_input = keras.random.normal(shape=(3, 5))
example_output = call_func(example_input)