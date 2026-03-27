import keras
import numpy as np

def call_func(inputs):
    return keras.ops.size(inputs)

x = keras.random.normal(shape=(3, 4, 5))
example_output = call_func(x)