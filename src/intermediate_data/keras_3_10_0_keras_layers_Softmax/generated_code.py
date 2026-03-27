import keras
import numpy as np

def call_func(axis=-1, inputs=None, mask=None):
    softmax_layer = keras.layers.Softmax(axis=axis)
    return softmax_layer(inputs, mask=mask)

# Generate random tensor with shape (batch_size, features)
input_tensor = np.random.randn(2, 5).astype(np.float32)
example_output = call_func(axis=-1, inputs=input_tensor)