import keras
import numpy as np

def call_func(inputs, axis=None, keepdims=False):
    return keras.ops.argmin(x=inputs, axis=axis, keepdims=keepdims)

random_tensor = keras.random.normal(shape=(3, 4, 5))
example_output = call_func(inputs=random_tensor, axis=1, keepdims=False)