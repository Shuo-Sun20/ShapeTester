import keras
import numpy as np

def call_func(inputs, x_factor=0.0, y_factor=0.0, interpolation="bilinear", fill_mode="reflect", fill_value=0.0, data_format=None, seed=None):
    layer = keras.layers.RandomShear(x_factor, y_factor, interpolation, fill_mode, fill_value, data_format, seed)
    return layer(inputs)

random_input = np.random.rand(2, 32, 32, 3).astype(np.float32)
example_output = call_func(random_input, x_factor=0.2, y_factor=0.2)