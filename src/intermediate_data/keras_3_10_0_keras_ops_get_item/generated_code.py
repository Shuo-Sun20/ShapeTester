import keras
import numpy as np

def call_func(inputs, key):
    return keras.ops.get_item(inputs, key)

x = keras.random.normal((4, 3, 2))
key = (0, slice(None), 0)
example_output = call_func(x, key)