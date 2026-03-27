import keras
import numpy as np

def call_func(inputs):
    return keras.ops.less(inputs[0], inputs[1])

example_output = call_func([keras.random.normal((3, 4)), keras.random.normal((3, 4))])