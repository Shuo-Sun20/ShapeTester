import keras
import numpy as np

def call_func(inputs, axis1, axis2):
    return keras.ops.swapaxes(inputs, axis1, axis2)

tensor = keras.random.normal(shape=(3, 4, 5))
axis1 = 0
axis2 = 2
example_output = call_func(tensor, axis1, axis2)