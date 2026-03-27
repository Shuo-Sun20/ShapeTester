import keras
import numpy as np

def call_func(activation, inputs, name=None, dtype=None):
    layer = keras.layers.Activation(activation, name=name, dtype=dtype)
    return layer(inputs)

random_input = np.random.randn(4, 3, 2)
example_output = call_func("relu", random_input)