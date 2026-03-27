import keras
import numpy as np

def call_func(inputs, name=None):
    layer_instance = keras.layers.Multiply(name=name)
    return layer_instance(inputs)

x1 = np.random.rand(2, 3, 4).astype('float32')
x2 = np.random.rand(2, 3, 4).astype('float32')
example_output = call_func([x1, x2])