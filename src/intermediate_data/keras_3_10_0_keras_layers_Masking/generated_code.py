import keras
import numpy as np

def call_func(mask_value, inputs):
    mask_layer = keras.layers.Masking(mask_value=mask_value)
    return mask_layer(inputs)

samples, timesteps, features = 32, 10, 8
inputs = np.random.random([samples, timesteps, features]).astype(np.float32)
inputs[:, 3, :] = 0.
inputs[:, 5, :] = 0.
example_output = call_func(mask_value=0.0, inputs=inputs)