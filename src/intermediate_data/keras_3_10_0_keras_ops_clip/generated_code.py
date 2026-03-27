import keras
import numpy as np

def call_func(inputs, x_min, x_max):
    x = inputs[0] if isinstance(inputs, list) and len(inputs) == 1 else inputs
    return keras.ops.clip(x, x_min, x_max)

example_output = call_func([keras.random.uniform((3, 4), -2, 2)], -1.0, 1.0)