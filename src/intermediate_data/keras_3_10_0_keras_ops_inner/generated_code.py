import keras
import numpy as np

def call_func(inputs):
    x1, x2 = inputs
    return keras.ops.inner(x1, x2)

x1 = keras.random.normal(shape=(3, 4, 5))
x2 = keras.random.normal(shape=(5,))
example_output = call_func([x1, x2])