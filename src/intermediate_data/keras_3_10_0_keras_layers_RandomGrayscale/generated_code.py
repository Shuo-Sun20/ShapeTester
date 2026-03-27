import keras
import numpy as np

def call_func(factor, data_format, seed, inputs):
    layer = keras.layers.RandomGrayscale(factor=factor, data_format=data_format, seed=seed)
    output = layer(inputs)
    return output

example_input = np.random.rand(4, 32, 32, 3).astype(np.float32)
example_output = call_func(factor=0.5, data_format='channels_last', seed=42, inputs=example_input)