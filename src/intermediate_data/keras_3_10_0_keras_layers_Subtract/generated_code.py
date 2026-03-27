import keras
import numpy as np

def call_func(inputs, name=None):
    subtract_layer = keras.layers.Subtract(name=name)
    return subtract_layer(inputs)

np.random.seed(42)
tensor1 = np.random.rand(2, 3, 4).astype(np.float32)
tensor2 = np.random.rand(2, 3, 4).astype(np.float32)
example_output = call_func([tensor1, tensor2])