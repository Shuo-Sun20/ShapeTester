import keras
import numpy as np

def call_func(inputs):
    x, y = inputs[0], inputs[1]
    return keras.ops.bitwise_and(x, y)

x = keras.random.randint(shape=(3, 4), minval=0, maxval=10)
y = keras.random.randint(shape=(3, 4), minval=0, maxval=10)
example_output = call_func([x, y])