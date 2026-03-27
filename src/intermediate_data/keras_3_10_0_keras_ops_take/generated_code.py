import keras
import numpy as np

def call_func(inputs, axis):
    x, indices = inputs
    return keras.ops.take(x, indices, axis)

x = keras.random.normal((3, 4, 5))
indices = keras.ops.convert_to_tensor([0, 2])
axis = 1
example_output = call_func([x, indices], axis)