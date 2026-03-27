import numpy as np
import keras

def call_func(cropping, inputs):
    cropping_layer = keras.layers.Cropping1D(cropping)
    if isinstance(inputs, list):
        return cropping_layer(*inputs)
    else:
        return cropping_layer(inputs)

# Generate random input tensor with shape (batch_size=4, sequence_length=10, features=5)
input_tensor = np.random.randn(4, 10, 5)
example_output = call_func(cropping=2, inputs=input_tensor)