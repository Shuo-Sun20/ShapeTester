import keras
import numpy as np

def call_func(target_shape, inputs):
    layer = keras.layers.Reshape(target_shape)
    return layer(inputs)

# Generate random input tensor with batch dimension
batch_size = 2
input_tensor = keras.random.normal((batch_size, 12))
# Call function and save output
example_output = call_func(target_shape=(3, 4), inputs=input_tensor)