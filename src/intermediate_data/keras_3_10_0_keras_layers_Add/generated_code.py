import keras
import numpy as np

def call_func(inputs, name=None):
    add_layer = keras.layers.Add(name=name)
    return add_layer(inputs)

# Generate random tensors
x1 = np.random.rand(2, 3, 4).astype('float32')
x2 = np.random.rand(2, 3, 4).astype('float32')
inputs = [x1, x2]

# Call function
example_output = call_func(inputs)