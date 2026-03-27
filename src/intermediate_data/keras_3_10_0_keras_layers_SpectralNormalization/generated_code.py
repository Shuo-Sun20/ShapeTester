import keras
import numpy as np

def call_func(layer, inputs, power_iterations=1):
    sn_layer = keras.layers.SpectralNormalization(layer=layer, power_iterations=power_iterations)
    if isinstance(inputs, list):
        output = sn_layer(*inputs)
    else:
        output = sn_layer(inputs)
    return output

# Construct valid input
x = np.random.randn(1, 10, 10, 1).astype('float32')
conv_layer = keras.layers.Conv2D(filters=2, kernel_size=2)
example_output = call_func(layer=conv_layer, inputs=x, power_iterations=1)