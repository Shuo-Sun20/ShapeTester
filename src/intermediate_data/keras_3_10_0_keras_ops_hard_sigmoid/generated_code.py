import keras
import numpy as np

def call_func(inputs):
    return keras.ops.hard_sigmoid(inputs)

example_input = np.random.randn(5)
example_output = call_func(example_input)