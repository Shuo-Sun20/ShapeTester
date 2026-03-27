import keras
import numpy as np

def call_func(inputs, k=0):
    if isinstance(inputs, list):
        x = inputs[0]
    else:
        x = inputs
    return keras.ops.triu(x, k)

x = keras.random.normal(shape=(4, 5, 6))
example_output = call_func(x, k=1)