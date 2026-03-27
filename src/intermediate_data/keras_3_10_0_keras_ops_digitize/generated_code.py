import keras
import numpy as np

def call_func(inputs):
    x, bins = inputs
    return keras.ops.digitize(x, bins)

x = keras.random.uniform(shape=(4,), minval=0.0, maxval=5.0)
bins = keras.ops.sort(keras.random.uniform(shape=(3,), minval=0.0, maxval=5.0))
example_output = call_func([x, bins])