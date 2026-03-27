import keras
import numpy as np

def call_func(inputs, negative_slope=0.2):
    return keras.ops.leaky_relu(x=inputs, negative_slope=negative_slope)

x = np.random.uniform(-1, 1, size=(5, 3))
example_output = call_func(x)