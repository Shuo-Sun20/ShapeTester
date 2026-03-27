import keras
import numpy as np

def call_func(axis, inputs):
    concat_layer = keras.layers.Concatenate(axis=axis)
    return concat_layer(inputs)

x = np.random.randn(2, 2, 5)
y = np.random.randn(2, 1, 5)
example_output = call_func(axis=1, inputs=[x, y])