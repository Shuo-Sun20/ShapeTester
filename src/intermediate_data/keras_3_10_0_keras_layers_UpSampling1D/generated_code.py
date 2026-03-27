import keras
import numpy as np

def call_func(size, inputs):
    layer = keras.layers.UpSampling1D(size=size)
    output = layer(inputs)
    return output

# Generate a random input tensor with shape (batch_size, steps, features)
batch_size = 2
steps = 3
features = 4
example_input = np.random.randn(batch_size, steps, features).astype(np.float32)

# Call the function
example_output = call_func(size=2, inputs=example_input)