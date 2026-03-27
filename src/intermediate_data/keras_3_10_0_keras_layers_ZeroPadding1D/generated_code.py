import numpy as np
import keras

def call_func(inputs, padding=1, data_format=None):
    layer = keras.layers.ZeroPadding1D(padding=padding, data_format=data_format)
    return layer(inputs)

# Construct a valid input tensor with shape (batch_size, axis_to_pad, features)
input_tensor = np.random.randn(2, 5, 3)  # batch_size=2, temporal_steps=5, features=3
example_output = call_func(input_tensor, padding=2, data_format='channels_last')