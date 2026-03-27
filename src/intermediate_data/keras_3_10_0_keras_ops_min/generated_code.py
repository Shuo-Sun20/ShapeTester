import keras
import numpy as np

def call_func(inputs, axis=None, keepdims=False, initial=None):
    return keras.ops.min(x=inputs, axis=axis, keepdims=keepdims, initial=initial)

random_tensor = keras.random.normal(shape=(3, 4))
example_output = call_func(random_tensor)