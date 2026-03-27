import keras
import numpy as np

def call_func(inputs, shift, axis=None):
    return keras.ops.roll(inputs, shift, axis)

random_tensor = keras.random.normal((3, 4, 5))
example_output = call_func(random_tensor, shift=2, axis=1)