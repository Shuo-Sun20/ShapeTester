import keras
import numpy as np

def call_func(inputs, condition=None, x1=None, x2=None):
    if isinstance(inputs, list):
        condition, x1, x2 = inputs
    return keras.ops.where(condition, x1, x2)

condition = keras.random.normal((3, 3)) > 0
x1 = keras.random.normal((3, 3))
x2 = keras.random.normal((3, 3))
example_output = call_func([condition, x1, x2])