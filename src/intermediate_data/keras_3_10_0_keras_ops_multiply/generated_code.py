import keras
import numpy as np

def call_func(inputs):
    x1, x2 = inputs[0], inputs[1]
    return keras.ops.multiply(x1, x2)

x1 = keras.random.normal(shape=(3, 4))
x2 = keras.random.normal(shape=(3, 4))
example_output = call_func([x1, x2])