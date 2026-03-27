import keras
import numpy as np

def call_func(inputs, axes, normalize=False):
    dot_layer = keras.layers.Dot(axes=axes, normalize=normalize)
    return dot_layer(inputs)

# Generate random input tensors
x = np.random.randn(2, 3, 5)
y = np.random.randn(2, 10, 3)
example_output = call_func([x, y], axes=(1, 2), normalize=False)